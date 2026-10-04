"""Shared physical decode runtime; later milestones enable prefetch and overlap.

This module owns integration, controller.py owns logical state, and cache.py owns
C+P role mappings. The old runtime remains the diagnostic parity reference.
"""
import hashlib,pickle,time
import numpy as np
import torch
import torch.distributed as dist
from la_physical_worker import LiveRuntime,EB,device_layout,nvtx_phase,plan_layout,PinnedH2DCache
from .runtime import gather_global_routes
from .controller import DecodePrefetchController
from .predictor import TransitionPredictor
from .compact_metadata import CompactMetadata

class DecodeOffloadRuntime(LiveRuntime):
 def __init__(self,a,model,backing,experts):
  self.predictor=TransitionPredictor(np.load('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/predictor/transition.npy'))
  super().__init__(a,model,backing,experts)
  self.cache=torch.empty((self.cap+a.arena_budget,EB//2),dtype=torch.bfloat16,device='cuda');self.keys=np.full(self.cap+a.arena_budget,-1,np.int32)
  self.h2d=PinnedH2DCache(self.cache,2)
  self.metadata=CompactMetadata(len(self.arrays['decode_origins'])//8)
 def reset(self):
  super().reset()
  self.controller=DecodePrefetchController(self.args.capacities,self.args.arena_budget,self.args.policy,self.args.seed,self.predictor)
  self.policy=self.controller.main;self.arena=self.controller.arena
  self.prefill_boundary=None;self.controller_times=[];self.debug_plan_checks=0
  if hasattr(self,'metadata'):self.metadata.calls=0
 def plan_event(self,layer,selected,weights,probs):
  with nvtx_phase('moe.metadata'):
   if self.index<48:g=gather_global_routes(layer,selected,weights,probs)
   else:g=self.metadata.collect(self.index,selected,probs,self.arrays['gates'][self.index])
  r=g.routes
  with nvtx_phase('moe.current_controller'):
   start=time.perf_counter()
   out,promotions,discards=self.controller.plan_current(self.index,r.selected_experts,r.routing_weights,r.origin_ranks,self.arrays['gates'][self.index])
   if self.index>=48:self.controller_times.append((time.perf_counter()-start)*1000)
   targets,effective,masses,lengths,destinations,fetches,row=out
   assert not promotions and not discards,'prefetch integration starts at M9'
   assert [list(f) for f in fetches]==self.reference[self.index]
   self.current_global_fetch_count=len(fetches)
   e=plan_layout(effective,lengths,destinations,r.origin_ranks,g.counts,self.rank)
   e.update(layer=layer,targets=targets,selected=selected.cpu().numpy(),fetches=[(key,int(self.arena.main_physical[rank][slot]),victim,rep) for rank,key,slot,victim,rep in fetches if rank==self.rank])
   e['groups']=[(expert,rows,cols,int(self.arena.main_physical[self.rank][np.flatnonzero(self.policy.slots[self.rank]==layer*128+expert)[0]])) for expert,rows,cols in e['groups']]
  if getattr(self.args,'debug_plan',False):
   payload=(fetches,self.policy.owner,self.policy.slots,self.arena.main_physical,self.arena.prefetch_physical)
   digest=np.frombuffer(hashlib.sha256(pickle.dumps(payload,protocol=4)).digest(),np.uint8).copy()
   tensor=torch.tensor(digest,device='cuda');all_hashes=torch.empty((8,32),dtype=torch.uint8,device='cuda')
   dist.all_gather_into_tensor(all_hashes.view(-1),tensor);assert bool((all_hashes==all_hashes[0]).all())
   self.debug_plan_checks+=1
  if self.args.phase!='MEASURE' and self.index%768==0:print(f'{self.args.policy} event={self.index} split_controller',flush=True)
  return device_layout(e)
 def execute(self,*args):
  result=super().execute(*args)
  if self.index==48:
   self.arena.assert_consistent();assert not self.arena.reservations and np.all(self.keys[self.cap:]<0)
   self.prefill_boundary=dict(main_roles=self.cap,prefetch_roles=self.args.arena_budget,prefetch_empty=True,main_resident=int(np.count_nonzero(self.keys[:self.cap]>=0)))
  return result
