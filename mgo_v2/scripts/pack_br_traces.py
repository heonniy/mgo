#!/usr/bin/env python3
"""Repack one master into R/B workloads; preserve exact W128 router history."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from mgo_v2.eviction import GateHistory
ROOT=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
P=Path(__file__).resolve().parents[1]/'experiments/br_ca_carep_cpu_headroom_20261003'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def main(name):
    audit=P/(name+'_horizon_audit.json');assert json.loads(audit.read_text())['status']=='PASS'
    requests=json.loads((ROOT/(name+'_requests.json')).read_text())['requests'];sources=[]
    for rank in range(8):
        path=ROOT/'captures'/name/f'rank{rank}'
        arrays={key:np.load(path/(key+'.npy'),mmap_mode='r') for key in ('prefill_selected','prefill_weights','prefill_router','decode_selected','decode_weights','decode_router')}
        lengths=[len(r['input_ids']) for r in requests[rank::8]];arrays['offsets']=np.cumsum([0]+lengths);sources.append(arrays)
    receipts=[]
    for world in (4,8):
        for batch in (8,16,32,64):
            dest=ROOT/'packed'/name/f'R{world}_B{batch}';done=dest/'receipt.json'
            if done.exists():
                old=json.loads(done.read_text());assert old['status']=='PASS' and all(sha(dest/r['name'])==r['sha256'] for r in old['files']);receipts.append(old);continue
            dest.mkdir(parents=True,exist_ok=False);n=world*batch;order=[i for rank in range(world) for i in range(rank,n,world)]
            lengths=[len(requests[i]['input_ids']) for i in order];prefill=sum(lengths)
            origins0=np.repeat(np.arange(world,dtype=np.int8).repeat(batch),lengths);origins=np.arange(world,dtype=np.int8).repeat(batch)
            offsets=np.cumsum(np.array([0]+[prefill]*48+[n]*(256*48),dtype=np.int64));total=int(offsets[-1]);np.save(dest/'offsets.npy',offsets);np.save(dest/'prefill_origins.npy',origins0);np.save(dest/'decode_origins.npy',origins);np.save(dest/'request_ids.npy',np.array(order,dtype=np.int32))
            selected=np.lib.format.open_memmap(dest/'selected.npy',mode='w+',dtype='uint8',shape=(total,8));weights=np.lib.format.open_memmap(dest/'weights.npy',mode='w+',dtype='float32',shape=(total,8));gates=np.lib.format.open_memmap(dest/'gates.npy',mode='w+',dtype='float32',shape=(257*48,128))
            history=GateHistory(48,128,128)
            for layer in range(48):
                pos=int(offsets[layer]);tail=[]
                for req,length in zip(order,lengths):
                    src=sources[req%8];lo,hi=src['offsets'][req//8:req//8+2]
                    selected[pos:pos+length]=src['prefill_selected'][layer,lo:hi];weights[pos:pos+length]=src['prefill_weights'][layer,lo:hi];pos+=length
                    tail.append(src['prefill_router'][layer,max(lo,hi-128):hi])
                    if sum(len(x) for x in tail)>256:tail=[np.concatenate(tail)[-128:]]
                history.update(layer,np.concatenate(tail)[-128:]);gates[layer]=[history.score(layer,e) for e in range(128)]
            for step in range(256):
                for layer in range(48):
                    event=(step+1)*48+layer;lo=int(offsets[event]);probs=np.empty((n,128),dtype=np.float32)
                    for j,req in enumerate(order):
                        src=sources[req%8];idx=(step,layer,req//8);selected[lo+j]=src['decode_selected'][idx];weights[lo+j]=src['decode_weights'][idx];probs[j]=src['decode_router'][idx]
                    history.update(layer,probs);gates[event]=[history.score(layer,e) for e in range(128)]
            for a in (selected,weights,gates):a.flush()
            receipt=dict(status='PASS',dataset=name,world=world,local_batch=batch,requests=n,horizons=[64,256],prefill_tokens=prefill,decode_steps=256,event_count=257*48,request_order=order,order_rule='rank-major; request i -> i%R; within rank ascending request ID',W128='exact production GateHistory on active full float32 router probabilities',source_audit_sha256=sha(audit),files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(dest.glob('*.npy'))])
            done.write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt);print(name,world,batch,'packed',flush=True)
            del selected,weights,gates
    (P/(name+'_pack_receipts.json')).write_text(json.dumps(receipts,indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['MATH','ShareGPT'],required=True);main(parser.parse_args().dataset)
