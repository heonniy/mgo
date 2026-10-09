"""Freeze larger distinct-input candidates and the previous-winner control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from pcie_host import write
from prepare_pcie_maxgain_candidates import main as candidates,sha,FIXED

WINNER=Path('/data2/esjung/mgo-results/pcie_topology_ablation_20261009/maxgain64_search_attempt1/candidates_repaired/family_3.json')
WINNER_SHA='32431c31694c633583fcc20cc9bab56b8db5c8b271d8d0ba7856c3b4735b7506'


def prepare(pool,out):
    assert sha(WINNER)==WINNER_SHA
    receipt=json.loads((pool/'POOL.json').read_text());assert receipt['status']=='PASS'
    for name,info in receipt['artifacts'].items():assert sha(pool/name)==info['sha256']
    out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(line) for line in (pool/'requests.jsonl').open()]
    tokens=np.memmap(pool/'tokens.uint32',dtype='<u4',mode='r',shape=(len(rows),512))
    anchor=json.loads(WINNER.read_text())['requests'];assert len(anchor)==64
    original_warm=json.loads((FIXED/'R4_C30_B16_L512_O64_warmup.json').read_text())['requests']
    def digest(row):return hashlib.sha256(np.asarray(row['input_ids'],np.uint32).tobytes()).hexdigest()
    excluded_sources={r['source_row'] for r in anchor+original_warm}
    excluded_hashes={digest(r) for r in anchor+original_warm}
    warm=list(original_warm)
    for i in np.random.default_rng(20261010).permutation(len(rows)):
        r=rows[int(i)]
        if r['source_row'] in excluded_sources or r['input_ids_uint32_sha256'] in excluded_hashes:continue
        warm.append(dict(r,request_id=r['source_row'],input_ids=np.asarray(tokens[i]).astype(int).tolist()))
        excluded_sources.add(r['source_row']);excluded_hashes.add(r['input_ids_uint32_sha256'])
        if len(warm)==256:break
    assert len(warm)==256
    # Distinct requests from the same original family first, then deterministic
    # whole-corpus fill. Preserve all original64 in their original input order.
    members=[i for i,r in enumerate(rows) if r['conversation_key']==anchor[0]['conversation_key']]
    order=members+np.random.default_rng(20261011).permutation(len(rows)).tolist()
    controls=list(anchor);used_sources={r['source_row'] for r in controls};used_hashes={digest(r) for r in controls}
    for i in order:
        r=rows[int(i)]
        if r['source_row'] in excluded_sources or r['source_row'] in used_sources or r['input_ids_uint32_sha256'] in excluded_hashes or r['input_ids_uint32_sha256'] in used_hashes:continue
        controls.append(dict(r,request_id=r['source_row'],input_ids=np.asarray(tokens[i]).astype(int).tolist()))
        used_sources.add(r['source_row']);used_hashes.add(r['input_ids_uint32_sha256'])
        if len(controls)==256:break
    assert len(controls)==256
    specs={}
    for batch in (16,32,64):
        root=out/f'B{batch}';root.mkdir();count=batch*4
        def manifest(name,source):
            requests=[dict(r,global_index=j,origin_rank=j//batch) for j,r in enumerate(source[:count])]
            assert len({r['source_row'] for r in requests})==count and len({digest(r) for r in requests})==count
            path=root/(name+'.json');write(path,dict(local_batch=batch,global_requests=count,input_tokens=512,requests=requests))
            return path
        warm_path=manifest('warmup',warm);control=manifest('previous_best_control',controls)
        candidates(pool,root/'candidates',batch,warm_path,control,'previous_best_control')
        spec_path=root/'candidates/CANDIDATES.json';spec=json.loads(spec_path.read_text())
        signatures=set();warm_sources={r['source_row'] for r in warm[:count]};warm_hashes={digest(r) for r in warm[:count]}
        for c in spec['candidates']:
            rr=json.loads(Path(c['path']).read_text())['requests'];h=hashlib.sha256(np.asarray([r['input_ids'] for r in rr],np.uint32).tobytes()).hexdigest()
            assert h not in signatures;signatures.add(h)
            assert not warm_sources & {r['source_row'] for r in rr} and not warm_hashes & {digest(r) for r in rr}
        specs[str(batch)]=dict(path=str(spec_path),sha256=sha(spec_path),candidates=len(signatures),
                            warmup_sha256=sha(warm_path),previous_best_control_sha256=sha(control),
                            original64_input_order_exact=[r['input_ids'] for r in controls[:64]]==[r['input_ids'] for r in anchor])
    registration=dict(status='FROZEN',source_pool=str(pool),source_pool_sha256=sha(pool/'POOL.json'),
        previous_winner_path=str(WINNER),previous_winner_sha256=WINNER_SHA,batches=specs,
        matrix=[dict(local_batch=b,global_requests=b*4,output_tokens=n,decode_forwards=n-1)
                for b in (16,32,64) for n in (64,128,256) if (b,n)!=(16,64)],
        historical_B16_O64='Retained original five-repeat result; no relabeling or rerun',
        control_semantics='First64 inputs retained. Larger batches add distinct inputs and repartition by new local batch; not the original rank-routing trace',
        search='Whole frozen corpus, 32 lexical/random/family candidates plus prior-winner control; remove an exact duplicate control batch before nomination',
        nomination='16-output G routing capture with exact native C++ independent R/G cold-cache replay once per local batch',
        screen='Top8 proxy nominations plus prespecified previous-winner control, fresh full requested output R/G pair per matrix cell',
        final='Top3 screen candidates plus control, counterordered three repeats; all finalist arms get two extras if any range/median exceeds5%; separate diagnostics',
        no_global_optimum_claim=True,raw_manifests_stay_under_data2=True)
    write(out/'REGISTRATION.json',registration)
    return registration


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.pool,a.out)),flush=True)
