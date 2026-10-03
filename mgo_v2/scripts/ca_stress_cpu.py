"""Routing-only balanced proxy and unchanged exact Gate replay inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib,resource
import numpy as np
from numba import njit
from mgo_v2.eviction import GateHistory
from br_carep_cpu import balanced_assignment,replay
from run_br_cpu import summarize
from ca_stress_common import *

@njit(cache=True)
def proxy_scores(routes,sample,orders,world,batch):
 result=np.zeros(len(orders),np.float64)
 for trial in range(len(orders)):
  for event in range(len(routes)):
   demand=np.zeros((128,world),np.int64)
   for pos in range(len(sample)):
    req=sample[orders[trial,pos]];rank=pos//batch
    for k in range(8):demand[routes[event,req,k],rank]+=1
   totals=demand.sum(1);active=np.flatnonzero(totals>0);m=len(active);quota=np.array([m//world+(r<m%world) for r in range(world)])
   # Realizable greedy balanced assignment, not a cached-policy replay or an
   # assertion that the proxy reaches the Hungarian optimum.
   gaps=np.zeros(m,np.int64)
   for j in range(m):
    values=np.sort(demand[active[j]]);gaps[j]=values[-1]-values[-2]
   order=np.argsort(-gaps,kind='mergesort');local=0.;baseline=0.
   for j in range(m):
    for rank in range(world):baseline+=demand[active[j],rank]*quota[rank]/m
   remaining=quota.copy()
   for jj in order:
    expert=active[jj];best=-1;value=-1
    for rank in range(world):
     if remaining[rank]>0 and demand[expert,rank]>value:best=rank;value=demand[expert,rank]
    assert best>=0;remaining[best]-=1;local+=value
   assert remaining.sum()==0
   result[trial]+=local-baseline
 return result/len(routes)

class Pool:
 def __init__(self,name):
  self.name=name;self.path=ROOT/'pool'/name;self.receipt=json.loads((self.path/'receipt.json').read_text());assert self.receipt['status']=='PASS'
  self.a={key:np.load(self.path/(key+'.npy'),mmap_mode='r') for key in ['lengths','offsets','decode_selected','decode_weights','decode_router','prefill_selected','prefill_weights','prefill_router_tail','proxy_routes']};self.n=self.receipt['pool_size']
 def candidate(self,world,batch,sample_seed,dp_seed):
  sample=np.random.default_rng(sample_seed).permutation(self.n)[:world*batch];order=np.random.default_rng(dp_seed).permutation(len(sample));return sample[order]
 def pack(self,order,world,batch):
  a=self.a;n=len(order);assert n==world*batch and len(set(order))==n
  lengths=a['lengths'][order];prefill=int(lengths.sum());offsets=np.cumsum(np.array([0]+[prefill]*48+[n]*256*48,np.int64));total=int(offsets[-1])
  selected=np.empty((total,8),np.uint8);weights=np.empty((total,8),np.float32);gates=np.empty((257*48,128),np.float32)
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
  for step in range(256):
   for layer in range(48):
    event=(step+1)*48+layer;lo=offsets[event];selected[lo:lo+n]=a['decode_selected'][step,layer,order];weights[lo:lo+n]=a['decode_weights'][step,layer,order]
    history.update(layer,a['decode_router'][step,layer,order[-128:]])
    gates[event]=(history.sums[layer]/len(history.rows[layer])).astype(np.float32)
  return dict(selected=selected,weights=weights,offsets=offsets,prefill_origins=origins0,decode_origins=origins,gates=gates)

def exact(arrays,world,cache,policy,seed):
 slots=48*128*cache//100;cap=np.array([slots//world+(r<slots%world) for r in range(world)],np.int64);similarity=np.zeros((48,128,128),np.float32)
 result=replay(arrays['selected'],arrays['weights'],arrays['offsets'],arrays['prefill_origins'],arrays['decode_origins'],arrays['gates'],similarity,cap,256,True,False,0 if policy=='BR' else 1,seed)
 rows,fetches=result[:2];assert np.max(fetches.max(1)-fetches.min(1))<=1
 assert np.all(rows[:,26]==0) and int(np.count_nonzero(result[2]>=0))<=slots
 state=hashlib.sha256()
 for array in result[2:]:state.update(array.tobytes())
 return dict(policy=policy,seed=seed,full=summarize(rows,fetches,slots,256),decode=summarize(rows[48:],fetches[48:],slots,256),final_state_sha256=state.hexdigest(),quota_sha256=hashlib.sha256(fetches.tobytes()).hexdigest(),quota_max_minus_min=int((fetches.max(1)-fetches.min(1)).max()),cache_capacity=slots)

def candidate_job(pool,job):
 start=time.monotonic();world=job['world'];batch=job['batch'];order=pool.candidate(world,batch,job['sample_seed'],job['dp_seed'])
 manifest=dict(dataset=pool.name,world=world,batch=batch,sample_seed=job['sample_seed'],dp_seed=job['dp_seed'],request_ids=order.tolist(),ranks=[order[r*batch:(r+1)*batch].tolist() for r in range(world)])
 key=f"{pool.name}_R{world}_B{batch}_s{job['sample_seed']}_d{job['dp_seed']}";path=ROOT/'candidates'/(key+'.json')
 if path.exists():old=json.loads(path.read_text());assert old['status']=='PASS' and old['manifest']==manifest and old['pool_receipt_sha256']==sha(pool.path/'receipt.json') and old['policy_source_sha256']==sha(PACKAGE/'scripts/br_carep_cpu.py');return dict(id=key,seconds=0,reused=True)
 arrays=pool.pack(order,world,batch)
 results=[]
 for cache in [30,40,50,60]:
  ca=exact(arrays,world,cache,'CA',42);br=[exact(arrays,world,cache,'BR',seed) for seed in SEEDS]
  results.append(dict(cache=cache,CA=ca,BR=br))
 out=dict(status='PASS',id=key,manifest=manifest,manifest_sha256=digest(manifest),request_subset_sha256=digest(sorted(order.tolist())),pool_receipt_sha256=sha(pool.path/'receipt.json'),route_sha256=hashlib.sha256(arrays['selected'].tobytes()).hexdigest(),gate_sha256=hashlib.sha256(arrays['gates'].tobytes()).hexdigest(),results=results,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,policy_source_sha256=sha(PACKAGE/'scripts/br_carep_cpu.py'))
 write(path,out);return dict(id=key,seconds=out['seconds'],peak_rss_bytes=out['peak_rss_bytes'])
