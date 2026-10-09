"""Freeze 32 whole-pool lexical candidates plus immutable headline control."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from pcie_host import write

FIXED=Path('/data2/esjung/datasets/frozen_pcie_topology_20261009')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(pool,out):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    receipt=json.loads((pool/'POOL.json').read_text());assert receipt['status']=='PASS'
    for name,info in receipt['artifacts'].items():assert sha(pool/name)==info['sha256']
    out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line) for line in (pool/'requests.jsonl').open()]
    n=len(rows);assert n==receipt['eligible_conversations']
    tokens=np.memmap(pool/'tokens.uint32',dtype='<u4',mode='r',shape=(n,512))
    warm_path=FIXED/'R4_C30_B16_L512_O64_warmup.json';warm=json.loads(warm_path.read_text())['requests']
    warm_sources={r['source_row'] for r in warm}
    warm_hashes={hashlib.sha256(np.asarray(r['input_ids'],np.uint32).tobytes()).hexdigest() for r in warm}
    allowed=np.array([i for i,r in enumerate(rows) if r['source_row'] not in warm_sources and r['input_ids_uint32_sha256'] not in warm_hashes],np.int64)
    assert len(allowed)>=128
    # Offline lexical similarity is only candidate construction, not a claim
    # about routing, fetch latency or serving TPOT.
    features=np.zeros((n,256),np.float32)
    for start in range(0,n,1024):
        t=np.asarray(tokens[start:start+1024],np.uint64)
        bucket=((t*2654435761+(t>>3))%256).astype(np.int64)
        offset=np.arange(len(t),dtype=np.int64)[:,None]*256
        features[start:start+len(t)]=np.bincount((bucket+offset).ravel(),minlength=len(t)*256).reshape(-1,256)
    df=(features>0).sum(axis=0);features=np.log1p(features)*np.log((n+1)/(df+1)).astype(np.float32)
    features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    def pick(order,count,used=None):
        used=set() if used is None else set(used);result=[];sources=set()
        for i in order:
            i=int(i);r=rows[i];digest=r['input_ids_uint32_sha256']
            if r['source_row'] in warm_sources or digest in warm_hashes or digest in used or r['source_row'] in sources:continue
            result.append(i);used.add(digest);sources.add(r['source_row'])
            if len(result)==count:return result
        raise RuntimeError('Insufficient distinct eligible inputs for candidate')
    def neighbors(vector):
        scores=features[allowed]@vector
        return allowed[np.argsort(-scores,kind='stable')]
    candidates=[];signatures=set()
    def freeze(label,indices,method,extra=None):
        assert len(indices)==64 and len(set(indices))==64
        rr=[]
        for position,i in enumerate(indices):
            source=rows[i];ids=np.asarray(tokens[i]).astype(np.int64).tolist()
            assert len(ids)==512
            rr.append(dict(**source,request_id=source['source_row'],input_ids=ids,
                           global_index=position,origin_rank=position//16))
        assert len({r['source_row'] for r in rr})==64 and len({r['input_ids_uint32_sha256'] for r in rr})==64
        assert not {r['source_row'] for r in rr}&warm_sources
        signature=hashlib.sha256(np.asarray([r['input_ids'] for r in rr],np.uint32).tobytes()).hexdigest()
        if signature in signatures:
            # Deterministic fallback for coincident lexical neighborhoods;
            # decide entirely before any nomination or serving measurement.
            replacement_seed=100000+len(candidates)
            replacement=pick(np.random.default_rng(replacement_seed).permutation(allowed),64)
            return freeze(label,replacement,'uniform seeded fallback for duplicate candidate',
                          dict(original_method=method,replacement_seed=replacement_seed))
        signatures.add(signature)
        path=out/(label+'.json');write(path,dict(cell='R4_C30_B16_L512_O64',phase='target',candidate=label,requests=rr))
        candidates.append(dict(candidate=label,path=str(path),sha256=sha(path),method=method,
                ordered_input_tokens_uint32_sha256=signature,
                conversation_families=len({r['conversation_key'] for r in rr}),request_ids=[r['request_id'] for r in rr],**(extra or {})))
    for seed in range(100,108):
        order=np.random.default_rng(seed).permutation(allowed);freeze(f'random_{seed}',pick(order,64),'uniform whole-pool seeded random',dict(seed=seed))
    seeds=allowed[np.floor((np.arange(8)+.5)/8*len(allowed)).astype(int)]
    orders=[neighbors(features[int(seed)]) for seed in seeds]
    for j,order in enumerate(orders):freeze(f'coherent_{j}',pick(order,64),'cosine neighborhood of whole-corpus stratum seed',dict(seed_pool_index=int(seeds[j])))
    for j in range(8):
        first=pick(orders[j],32);used={rows[i]['input_ids_uint32_sha256'] for i in first}
        second=pick(orders[(j+4)%8],32,used)
        indices=first+second if j%2==0 else [i for pair in zip(first,second) for i in pair]
        freeze(f'mixed_{j}',indices,'two lexical neighborhoods; block/interleaved rank partitions',dict(partition='blocked' if j%2==0 else 'interleaved'))
    families=Counter(rows[int(i)]['conversation_key'] for i in allowed)
    for j,(key,count) in enumerate(families.most_common(8)):
        members=[int(i) for i in allowed if rows[int(i)]['conversation_key']==key]
        center=features[members].mean(axis=0);order=members+neighbors(center).tolist()
        freeze(f'family_{j}',pick(order,64),'largest real conversation family plus nearest distinct inputs',dict(family=key,family_pool_requests=count))
    assert len(candidates)==32
    control=FIXED/'R4_C30_B16_L512_O64_target.json'
    control_rows=json.loads(control.read_text())['requests']
    control_signature=hashlib.sha256(np.asarray([r['input_ids'] for r in control_rows],np.uint32).tobytes()).hexdigest()
    assert control_signature not in signatures,'Generated candidate duplicates the immutable control'
    candidates.insert(0,dict(candidate='reference_control',path=str(control),sha256=sha(control),method='immutable headline64 reference',request_ids=[r['request_id'] for r in control_rows]))
    spec=dict(status='FROZEN',source_pool=str(pool),source_pool_sha256=sha(pool/'POOL.json'),
              source_dataset_sha256=receipt['source_sha256'],eligible_pool=len(allowed),whole_dataset_rows=receipt['scanned_source_rows'],
              objective='Maximize physically measured G-NEAR vs R-NEAR full64 median TPOT reduction',
              global_requests=64,local_batch=16,input_tokens=512,final_output_tokens=64,nomination_output_tokens=16,
              placement='LA_CA_NEAR',expert_executor='grouped decode / native prefill',metadata='native C++',
              cache_slots=1843,physical_gpus=[0,1,4,5],warmup=dict(path=str(warm_path),sha256=sha(warm_path)),
              lexical_features='256 hashed token-count TF-IDF buckets; stable cosine order; offline nomination only',
              candidates=candidates,raw_tokens_stay_external=True,selection_timing_not_final_estimate=True)
    write(out/'CANDIDATES.json',spec);print(json.dumps(dict(status='FROZEN',candidates=len(candidates),out=str(out))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.pool,a.out)
