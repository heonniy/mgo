"""True one-partial-per-token/rank return, with explicit accumulation precision.

This remains a separately validated candidate. It does not claim legacy BF16
expert-order bitwise equivalence. Input contributions and ordering are checked
independently; physical model output/argmax differences must be reported.
"""
import torch

def combine_rank_partials(transport,hidden,parts,event,accumulation=torch.float32,unique_rows=False):
 if accumulation not in (torch.bfloat16,torch.float32,torch.float64):raise ValueError(accumulation)
 fast=unique_rows and accumulation==torch.bfloat16 and event.get('unique_combine_validated',False)
 if fast:
  from .unique_combine import add_unique_rows_
 partial=torch.zeros((sum(event['recv_counts']),hidden.shape[-1]),device=hidden.device,dtype=accumulation)
 for (_,rows,_,_),values in zip(event['groups'],parts):
  if fast:add_unique_rows_(partial,rows,values)
  else:partial.index_add_(0,rows,values.to(accumulation))
 returned,_=transport.exchange(partial,event['recv_counts'],event['send_counts'])
 transport.return_bytes+=(sum(event['recv_counts'])-event['recv_counts'][transport.rank])*hidden.shape[-1]*partial.element_size()
 output=torch.zeros_like(hidden,dtype=accumulation);offset=0
 for count in event['send_counts']:
  indices=event['send_idx'][offset:offset+count];values=returned[offset:offset+count]
  if fast:add_unique_rows_(output,indices,values)
  else:output.index_add_(0,indices,values)
  offset+=count
 return output.to(hidden.dtype)
