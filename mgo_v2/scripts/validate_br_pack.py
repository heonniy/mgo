#!/usr/bin/env python3
"""Independent full-window means and raw-route comparisons on packed inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
P=Path(__file__).resolve().parents[1]/'experiments/br_ca_carep_cpu_headroom_20261003'
def main(name):
    req=json.loads((ROOT/(name+'_requests.json')).read_text())['requests'];source=[]
    for rank in range(8):
        path=ROOT/'captures'/name/f'rank{rank}';source.append({k:np.load(path/(k+'.npy'),mmap_mode='r') for k in ('decode_selected','decode_weights','decode_router','prefill_router')});source[-1]['offsets']=np.cumsum([0]+[len(r['input_ids']) for r in req[rank::8]])
    checked=0
    for world in (4,8):
        for batch in (8,16,32,64):
            path=ROOT/'packed'/name/f'R{world}_B{batch}';order=np.load(path/'request_ids.npy');g=np.load(path/'gates.npy',mmap_mode='r');off=np.load(path/'offsets.npy');s=np.load(path/'selected.npy',mmap_mode='r');w=np.load(path/'weights.npy',mmap_mode='r');assert sorted(order.tolist())==list(range(world*batch))
            for layer in (0,47):
                history=np.concatenate([source[i%8]['prefill_router'][layer,source[i%8]['offsets'][i//8]:source[i%8]['offsets'][i//8+1]] for i in order])[-128:].astype(np.float64);assert np.array_equal(g[layer],history.mean(0).astype(np.float32));checked+=1
                for step in range(256):
                    probs=np.stack([source[i%8]['decode_router'][step,layer,i//8] for i in order]);history=np.concatenate((history,probs))[-128:];event=(step+1)*48+layer;assert np.array_equal(g[event],history.mean(0).astype(np.float32));checked+=1
                    if step in (0,63,64,255):
                        assert np.array_equal(s[off[event]:off[event+1]],np.stack([source[i%8]['decode_selected'][step,layer,i//8] for i in order]));assert np.array_equal(w[off[event]:off[event+1]],np.stack([source[i%8]['decode_weights'][step,layer,i//8] for i in order]))
            print(name,world,batch,'PASS',flush=True)
    result=dict(status='PASS',dataset=name,packed_configurations=8,independent_W128_checks=checked,sampled_layers=[0,47],selected_weight_comparisons_decode_steps=[1,64,65,256])
    dest=P/(name+'_pack_validation.json');assert not dest.exists(),'Reuse completed validation rather than overwriting it';dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['MATH','ShareGPT'],required=True);main(parser.parse_args().dataset)
