"""One fixed-record decode collective, preserving Gate W128 tail semantics."""
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist

class CompactMetadata:
 def __init__(self,batch,topk=8,experts=128,window=128):
  if batch<window:raise ValueError('compact primary path requires batch >= Gate window')
  self.batch=batch;self.topk=topk;self.experts=experts;self.window=window
  self.world=dist.get_world_size();self.rank=dist.get_rank();self.header_bytes=24
  self.ids_end=24+batch*topk;self.record_bytes=self.ids_end+experts*4
  self.send=torch.empty(self.record_bytes,dtype=torch.uint8,device='cuda')
  self.recv=torch.empty(self.world*self.record_bytes,dtype=torch.uint8,device='cuda')
  self.host=torch.empty(self.world*self.record_bytes,dtype=torch.uint8,pin_memory=True)
  self.origins=np.repeat(np.arange(self.world,dtype=np.int64),batch)
  # Admission uses expert IDs/counts, not weights, with substitution OFF.
  self.controller_weights=np.ones((self.world*batch,topk),np.float32)
  self.calls=0
 def collect(self,event,selected,probs,frozen_gate=None):
  if tuple(selected.shape)!=(self.batch,self.topk):raise ValueError('fixed-batch metadata shape mismatch')
  header=torch.tensor([1,event,self.batch],dtype=torch.int64,device=selected.device)
  self.send[:24].copy_(header.view(torch.uint8));self.send[24:self.ids_end].copy_(selected.to(torch.uint8).reshape(-1))
  if frozen_gate is None:
   # With B>=W, global rank-order Gate W128 consists only of the final rank tail.
   gate=(probs[-self.window:].to(torch.float64).sum(0)/self.window).to(torch.float32)
  else:gate=torch.as_tensor(frozen_gate,dtype=torch.float32,device=selected.device)
  self.send[self.ids_end:].copy_(gate.view(torch.uint8))
  dist.all_gather_into_tensor(self.recv,self.send);self.calls+=1
  self.host.copy_(self.recv);raw=self.host.numpy().reshape(self.world,self.record_bytes)
  headers=raw[:,:24].copy().view(np.int64).reshape(self.world,3)
  if not np.all(headers==np.array([1,event,self.batch])):raise RuntimeError('metadata header drift')
  ids=raw[:,24:self.ids_end].copy().reshape(self.world*self.batch,self.topk)
  if np.any(ids>=self.experts):raise RuntimeError('invalid expert ID')
  gate_scores=raw[-1,self.ids_end:].copy().view(np.float32)
  hist=np.bincount((ids.astype(np.int64)+self.origins[:,None]*self.experts).ravel(),minlength=self.world*self.experts).reshape(self.world,self.experts)
  routes=SimpleNamespace(selected_experts=ids,routing_weights=self.controller_weights,origin_ranks=self.origins,full_router_probs=None)
  return SimpleNamespace(routes=routes,counts=[self.batch]*self.world,local_offset=self.rank*self.batch,gate_scores=gate_scores,histogram=hist)
