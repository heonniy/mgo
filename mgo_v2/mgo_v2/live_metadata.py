"""Native routing metadata with exact global Gate W128 rolling history.

The frozen-trace compact path remains unchanged. This path transmits current
router probabilities only; no future routing or teacher tokens are available.
"""
from types import SimpleNamespace
import time
import numpy as np
import torch
import torch.distributed as dist

class LiveMetadata:
 def __init__(self,batch,history,topk=8,experts=128,layers=48):
  if not 1<=topk<=8 or not 1<=experts<=256 or layers<1:raise ValueError('unsupported routing geometry')
  self.batch=batch;self.history=history;self.topk=topk;self.experts=experts;self.layers=layers
  self.world=dist.get_world_size();self.calls=0
  self.tail=min(batch,128);self.ids_end=24+batch*topk;self.record_bytes=self.ids_end+self.tail*experts*4
  self.send=torch.empty(self.record_bytes,dtype=torch.uint8,device='cuda')
  self.recv=torch.empty(self.world*self.record_bytes,dtype=torch.uint8,device='cuda')
  self.host=torch.empty(self.world*self.record_bytes,dtype=torch.uint8,pin_memory=True)
  self.origins=np.repeat(np.arange(self.world,dtype=np.int64),batch)
  self.weights=np.ones((self.world*batch,topk),np.float32)
  self.profile=False;self.last_profile=None
 def collect(self,event,selected,probs,frozen_gate=None):
  assert frozen_gate is None and selected.shape==(self.batch,self.topk) and probs.shape==(self.batch,self.experts)
  started_ns=time.perf_counter_ns() if self.profile else None
  self.send[:24].copy_(torch.tensor([1,event,self.batch],device='cuda',dtype=torch.int64).view(torch.uint8))
  self.send[24:self.ids_end].copy_(selected.to(torch.uint8).reshape(-1))
  self.send[self.ids_end:].copy_(probs[-self.tail:].float().contiguous().view(torch.uint8).reshape(-1))
  packed_ns=time.perf_counter_ns() if self.profile else None
  dist.all_gather_into_tensor(self.recv,self.send)
  gathered_ns=time.perf_counter_ns() if self.profile else None
  self.host.copy_(self.recv);self.calls+=1
  copied_ns=time.perf_counter_ns() if self.profile else None
  raw=self.host.numpy().reshape(self.world,self.record_bytes)
  headers=raw[:,:24].copy().view(np.int64).reshape(self.world,3)
  assert np.all(headers==[1,event,self.batch])
  ids=raw[:,24:self.ids_end].copy().reshape(self.world*self.batch,self.topk)
  probs_all=raw[:,self.ids_end:].copy().view(np.float32).reshape(-1,self.experts)
  layer=event%self.layers;self.history.update(layer,probs_all)
  gates=(self.history.sums[layer]/max(1,len(self.history.rows[layer]))).astype(np.float32)
  hist=np.bincount((ids.astype(np.int64)+self.origins[:,None]*self.experts).ravel(),minlength=self.world*self.experts).reshape(self.world,self.experts)
  routes=SimpleNamespace(selected_experts=ids,routing_weights=self.weights,origin_ranks=self.origins,full_router_probs=None)
  if self.profile:
   finished_ns=time.perf_counter_ns()
   self.last_profile=dict(pack_submit_host_ms=(packed_ns-started_ns)/1e6,
    allgather_submit_host_ms=(gathered_ns-packed_ns)/1e6,
    recv_to_host_wait_ms=(copied_ns-gathered_ns)/1e6,
    parse_history_host_ms=(finished_ns-copied_ns)/1e6)
  return SimpleNamespace(routes=routes,counts=[self.batch]*self.world,histogram=hist,gate_scores=gates)
