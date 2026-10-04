"""10k role operations plus physical CUDA content/pointer checks when requested."""
import argparse,json
import numpy as np
from env_offload_policy import Policy
from mgo_v2.cache import SlotArena

def main():
 p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');p.add_argument('--output');a=p.parse_args()
 m=Policy([4]*8,np.zeros((48,128,128),np.float32),False,0,42);arena=SlotArena(m,2);rng=np.random.default_rng(55)
 storage=[np.full(6,-1,np.int64) for _ in range(8)];promotions=0;discards=0
 for tick in range(10000):
  rank=int(rng.integers(8));key=int(rng.integers(6144));pfslot=int(rng.integers(2))
  if m.owner[key]:continue
  physical=arena.reserve(rank,key,pfslot,key//128);storage[rank][physical]=key
  if rng.random()<.25:arena.discard(key);discards+=1
  else:
   slot=int(rng.integers(4));prom=arena.promote(key,slot,tick+1);assert prom.promoted_physical_slot==physical
   assert storage[rank][arena.main_physical[rank][slot]]==key;promotions+=1
  arena.assert_consistent()
 assert promotions>6000 and discards>1500
 gpu_checks=[]
 if a.cuda:
  import torch
  torch.cuda.set_per_process_memory_fraction(.15)
  m=Policy([2]*8,np.zeros((48,128,128),np.float32),False,0,42);arena=SlotArena(m,2)
  weights=torch.zeros((4,9437184//2),dtype=torch.bfloat16,device='cuda')
  stream=torch.cuda.Stream();done=torch.cuda.Event()
  for mode,key in [('READY',10),('QUEUED',11),('INFLIGHT',12)]:
   pfslot=0;physical=arena.reserve(0,key,pfslot,0);pointer=weights[physical].data_ptr()
   host=torch.full_like(weights[physical],float(key),device='cpu',pin_memory=True)
   if mode!='QUEUED':
    with torch.cuda.stream(stream):weights[physical].copy_(host,non_blocking=True);done.record(stream)
    if mode=='READY':done.synchronize()
   promotion=arena.promote(key,0,1)
   if mode=='QUEUED':
    with torch.cuda.stream(stream):weights[physical].copy_(host,non_blocking=True);done.record(stream)
   torch.cuda.current_stream().wait_event(done)
   assert weights[promotion.promoted_physical_slot].data_ptr()==pointer
   assert bool((weights[promotion.promoted_physical_slot]==key).all())
   arena.assert_consistent();gpu_checks.append(dict(state_at_promotion=mode,pointer_unchanged=True,content_valid=True,D2D_copies=0))
 result=dict(status='PASS',randomized_operations=10000,promotions=promotions,discards=discards,gpu_checks=gpu_checks)
 if a.output:
  from pathlib import Path
  Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
