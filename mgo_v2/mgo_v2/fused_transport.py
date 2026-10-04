"""Two payload collectives with coalesced token-to-rank forward packets.

Exact return mode retains each BF16 expert contribution so the source can add
in canonical expert order. Rank-partial compression is exposed diagnostically:
it changes BF16 association and must never be silently substituted for exact.
"""
from dataclasses import dataclass
import torch
import torch.distributed as dist

@dataclass
class ForwardPacket:
 raw: torch.Tensor
 work: object
 hidden_size: int
 dtype: torch.dtype
 def finish(self):
  if self.work is not None:self.work.wait();self.work=None
  n=self.raw.shape[0];hb=self.hidden_size*2
  hidden=self.raw[:,:hb].contiguous().view(self.dtype).view(n,self.hidden_size)
  weights=self.raw[:,hb:hb+16].contiguous().view(self.dtype).view(n,8)
  ids=self.raw[:,hb+16:hb+24].contiguous()
  return hidden,weights,ids

class FusedTokenRankTransport:
 def __init__(self,return_mode='exact'):
  if return_mode not in ('exact','rank-partial'):raise ValueError(return_mode)
  self.return_mode=return_mode;self.calls=0;self.forward_bytes=0;self.return_bytes=0;self.rank=dist.get_rank()
 def exchange(self,x,send,recv,async_op=False):
  y=torch.empty((sum(recv),)+tuple(x.shape[1:]),device=x.device,dtype=x.dtype)
  work=dist.all_to_all_single(y,x.contiguous(),output_split_sizes=recv,input_split_sizes=send,async_op=async_op);self.calls+=1
  return y,work
 def forward(self,hidden,dense,e,async_op=False):
  if hidden.dtype!=torch.bfloat16:raise ValueError('validated wire format requires BF16')
  idx=e['send_idx'];ids=e['send_eids'];weights=dense[idx[:,None],ids.clamp_min(0)]*(ids>=0)
  n=len(idx);h=hidden.shape[-1];hb=h*2
  payload=torch.empty((n,hb+24),device=hidden.device,dtype=torch.uint8)
  payload[:,:hb]=hidden[idx].contiguous().view(torch.uint8).view(n,hb)
  payload[:,hb:hb+16]=weights.contiguous().view(torch.uint8).view(n,16)
  payload[:,hb+16:]=ids.to(torch.uint8)
  raw,work=self.exchange(payload,e['send_counts'],e['recv_counts'],async_op)
  self.forward_bytes+=(sum(e['send_counts'])-e['send_counts'][self.rank])*(hb+24)
  return ForwardPacket(raw,work,h,hidden.dtype)
 def combine(self,hidden,parts,e):
  if self.return_mode=='exact':
   values=torch.cat(parts) if parts else hidden.new_empty((0,hidden.shape[-1]))
   returned,_=self.exchange(values[e['return_order']],e['return_counts'],e['return_recv_counts'])
   self.return_bytes+=(sum(e['return_counts'])-e['return_counts'][self.rank])*hidden.shape[-1]*2
   output=torch.zeros_like(hidden)
   for idx,pos in e['combine']:output.index_add_(0,idx,returned[pos])
   return output
  partial=hidden.new_zeros((sum(e['recv_counts']),hidden.shape[-1]))
  for (_,rows,_,_),values in zip(e['groups'],parts):partial.index_add_(0,rows,values)
  returned,_=self.exchange(partial,e['recv_counts'],e['send_counts'])
  self.return_bytes+=(sum(e['recv_counts'])-e['recv_counts'][self.rank])*hidden.shape[-1]*2
  output=torch.zeros_like(hidden);offset=0
  for count in e['send_counts']:
   output.index_add_(0,e['send_idx'][offset:offset+count],returned[offset:offset+count]);offset+=count
  return output
