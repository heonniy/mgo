"""CPU-only decode64 prefix input and frozen packet paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np
from ca_stress_cpu import Pool
from ca_stress_common import sha,digest,write,PACKAGE as P,PYTHON,SEEDS
from mgo_v2.eviction import GateHistory
ROOT=Path('/home/hwlee/mgo-results/fetch_matched_b32_a2a_20261004')
PACKET=P/'experiments/fetch_matched_b32_a2a_20261004'
class PrefixPool(Pool):
 def pack(self,order,world,batch):
  a=self.a;n=len(order);assert n==world*batch and len(set(order))==n
  lengths=a['lengths'][order];prefill=int(lengths.sum());offsets=np.cumsum(np.array([0]+[prefill]*48+[n]*64*48,np.int64));total=int(offsets[-1])
  selected=np.empty((total,8),np.uint8);weights=np.empty((total,8),np.float32);gates=np.empty((65*48,128),np.float32)
  origins=np.repeat(np.arange(world,dtype=np.int8),batch);origins0=np.repeat(origins,lengths);history=GateHistory(48,128,128)
  for layer in range(48):
   pos=offsets[layer]
   for req,length in zip(order,lengths):
    lo,hi=a['offsets'][req:req+2];selected[pos:pos+length]=a['prefill_selected'][layer,lo:hi];weights[pos:pos+length]=a['prefill_weights'][layer,lo:hi];pos+=length
   tail=[];count=0
   for req in reversed(order):
    length=min(int(a['lengths'][req]),128);tail.append(a['prefill_router_tail'][layer,req,:length]);count+=length
    if count>=128:break
   history.update(layer,np.concatenate(tail[::-1])[-128:]);gates[layer]=(history.sums[layer]/len(history.rows[layer])).astype(np.float32)
  for step in range(64):
   for layer in range(48):
    event=(step+1)*48+layer;lo=offsets[event];selected[lo:lo+n]=a['decode_selected'][step,layer,order];weights[lo:lo+n]=a['decode_weights'][step,layer,order]
    history.update(layer,a['decode_router'][step,layer,order[-128:]])
    gates[event]=(history.sums[layer]/len(history.rows[layer])).astype(np.float32)
  return dict(selected=selected,weights=weights,offsets=offsets,prefill_origins=origins0,decode_origins=origins,gates=gates)

