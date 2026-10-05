"""Byte preservation and pre-copy bounds validation for the native BF16 path."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from mgo_v2.pinned_h2d import copy_expert_to_stage

def main():
 torch.set_num_threads(1)
 # All BF16 bit patterns, including NaNs, are copied without conversion.
 raw=torch.arange(65536,dtype=torch.int32).to(torch.int16)
 backing=raw.view(torch.bfloat16)
 src=[backing[:1024],backing[1024:40000].view(192,203),backing[40000:]]
 for mode in ('torch','memmove'):
  dst=torch.empty_like(backing);assert copy_expert_to_stage(dst,src,mode)==65536
  assert torch.equal(dst.view(torch.int16),raw)
 dst=torch.full((12,),42,dtype=torch.bfloat16);initial=dst.clone()
 bad=[([torch.empty(13,dtype=torch.bfloat16)],'memmove'),([torch.empty(11,dtype=torch.bfloat16)],'memmove'),([torch.empty(12,dtype=torch.float32)],'memmove'),([torch.empty(24,dtype=torch.bfloat16)[::2]],'memmove'),([torch.empty(12,dtype=torch.bfloat16)],'invalid')]
 for src,mode in bad:
  try:copy_expert_to_stage(dst,src,mode)
  except ValueError:pass
  else:raise AssertionError('unsafe staging input accepted')
  assert torch.equal(dst,initial),'failure wrote destination'
 print('PASS all BF16 bit patterns, size/dtype/contiguity rejection, no write on invalid input')
if __name__=='__main__':main()
