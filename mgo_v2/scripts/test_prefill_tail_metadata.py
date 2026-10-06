"""Distributed CPU regression: optimized metadata retains exact global Gate history."""
import numpy as np
import torch
import torch.distributed as dist
from mgo_v2.runtime import gather_global_routes
from mgo_v2.eviction import GateHistory

def main():
 torch.set_num_threads(1);dist.init_process_group('gloo');rank=dist.get_rank()
 checks=0
 for counts in ([0,0,0,0],[0,1,2,3],[16,16,16,16],[128,0,0,0],[37,0,64,99],[4096]*4,[32768]*4):
  rng=np.random.default_rng(7);all_probs=rng.random((sum(counts),128),dtype=np.float32)
  start=sum(counts[:rank]);probs=torch.from_numpy(all_probs[start:start+counts[rank]])
  weights,ids=torch.topk(probs,8,dim=-1)
  full=gather_global_routes(0,ids,weights,probs)
  tail=gather_global_routes(0,ids,weights,probs,probability_tail=128)
  for field in ('selected_experts','routing_weights','origin_ranks'):
   np.testing.assert_array_equal(getattr(full.routes,field),getattr(tail.routes,field))
  np.testing.assert_array_equal(tail.routes.full_router_probs,all_probs[-128:])
  old,new=GateHistory(1,128),GateHistory(1,128)
  seed=rng.random((64,128),dtype=np.float32)
  old.update(0,seed);new.update(0,seed)
  old.update(0,full.routes.full_router_probs);new.update(0,tail.routes.full_router_probs)
  np.testing.assert_array_equal(old.sums,new.sums)
  np.testing.assert_array_equal(np.asarray(old.rows[0]),np.asarray(new.rows[0]));checks+=1
 if rank==0:print(f'PASS: {checks} distributed cases, including uneven/empty ranks and both full experiment sizes')
 dist.destroy_process_group()
if __name__=='__main__':main()
