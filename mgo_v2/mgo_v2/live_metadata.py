"""Native routing metadata with exact global Gate W128 rolling history.

The frozen-trace compact path remains unchanged. This path transmits current
router probabilities only; no future routing or teacher tokens are available.
"""
from types import SimpleNamespace
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
 def collect(self,event,selected,probs,frozen_gate=None):
  assert frozen_gate is None and selected.shape==(self.batch,self.topk) and probs.shape==(self.batch,self.experts)
  self.send[:24].copy_(torch.tensor([1,event,self.batch],device='cuda',dtype=torch.int64).view(torch.uint8))
  self.send[24:self.ids_end].copy_(selected.to(torch.uint8).reshape(-1))
  self.send[self.ids_end:].copy_(probs[-self.tail:].float().contiguous().view(torch.uint8).reshape(-1))
  dist.all_gather_into_tensor(self.recv,self.send);self.host.copy_(self.recv);self.calls+=1
  raw=self.host.numpy().reshape(self.world,self.record_bytes)
  headers=raw[:,:24].copy().view(np.int64).reshape(self.world,3)
  assert np.all(headers==[1,event,self.batch])
  ids=raw[:,24:self.ids_end].copy().reshape(self.world*self.batch,self.topk)
  probs_all=raw[:,self.ids_end:].copy().view(np.float32).reshape(-1,self.experts)
  layer=event%self.layers;self.history.update(layer,probs_all)
  gates=(self.history.sums[layer]/max(1,len(self.history.rows[layer]))).astype(np.float32)
  hist=np.bincount((ids.astype(np.int64)+self.origins[:,None]*self.experts).ravel(),minlength=self.world*self.experts).reshape(self.world,self.experts)
  routes=SimpleNamespace(selected_experts=ids,routing_weights=self.weights,origin_ranks=self.origins,full_router_probs=None)
  return SimpleNamespace(routes=routes,counts=[self.batch]*self.world,histogram=hist,gate_scores=gates)

 def collect_profiled(self,event,selected,probs,phase,frozen_gate=None):
  """Same decode packet as collect, split only for a separate diagnostic run."""
  assert frozen_gate is None and selected.shape==(self.batch,self.topk) and probs.shape==(self.batch,self.experts)
  with phase('metadata_gpu_packet_pack'):
   self.send[:24].copy_(torch.tensor([1,event,self.batch],device='cuda',dtype=torch.int64).view(torch.uint8))
   self.send[24:self.ids_end].copy_(selected.to(torch.uint8).reshape(-1))
   self.send[self.ids_end:].copy_(probs[-self.tail:].float().contiguous().view(torch.uint8).reshape(-1))
  with phase('metadata_rank_all_gather'):
   dist.all_gather_into_tensor(self.recv,self.send)
  with phase('metadata_device_to_host'):
   self.host.copy_(self.recv)
  self.calls+=1
  with phase('metadata_cpu_parse'):
   raw=self.host.numpy().reshape(self.world,self.record_bytes)
   headers=raw[:,:24].copy().view(np.int64).reshape(self.world,3)
   assert np.all(headers==[1,event,self.batch])
   ids=raw[:,24:self.ids_end].copy().reshape(self.world*self.batch,self.topk)
   probs_all=raw[:,self.ids_end:].copy().view(np.float32).reshape(-1,self.experts)
  with phase('metadata_gate_history'):
   layer=event%self.layers;self.history.update(layer,probs_all)
   gates=(self.history.sums[layer]/max(1,len(self.history.rows[layer]))).astype(np.float32)
  with phase('metadata_demand_histogram'):
   hist=np.bincount((ids.astype(np.int64)+self.origins[:,None]*self.experts).ravel(),minlength=self.world*self.experts).reshape(self.world,self.experts)
  routes=SimpleNamespace(selected_experts=ids,routing_weights=self.weights,origin_ranks=self.origins,full_router_probs=None)
  return SimpleNamespace(routes=routes,counts=[self.batch]*self.world,histogram=hist,gate_scores=gates)
