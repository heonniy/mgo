#!/usr/bin/env python3
"""Bounded R4 transfer calibration; identical all-to-all API as MoE dispatch."""
import argparse,json,os
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
boot=pin_rank_before_cuda_import()
if os.environ['MGO_TRANSPORT']!='T1':os.environ.pop('NCCL_P2P_DISABLE',None)
import numpy as np
import torch
import torch.distributed as dist

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 assert boot['visible_gpu']==str([0,1,4,5][boot['local_rank']])
 assert 'NCCL_SHM_DISABLE' not in os.environ
 if os.environ['MGO_TRANSPORT'].startswith('R'):assert os.environ.get('NCCL_P2P_LEVEL')=='LOC' and 'NCCL_P2P_DISABLE' not in os.environ
 else:assert 'NCCL_P2P_LEVEL' not in os.environ
 torch.set_num_threads(1);torch.cuda.set_device(0);dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 rank=dist.get_rank();assert dist.get_world_size()==4
 # Decode proxy: 8 token rows per peer for dispatch, 16 exact-expert rows
 # per peer for combine, hidden size 2048 and BF16. Self bytes excluded.
 sends=[torch.full((4*n,2048),float(rank),device='cuda',dtype=torch.bfloat16) for n in (8,16)]
 recvs=[torch.empty_like(t) for t in sends]
 cpu=torch.empty(9*1024**2,dtype=torch.uint8,pin_memory=True);cpu.fill_(17)
 gpu=torch.empty_like(cpu,device='cuda')
 hs,ps=torch.cuda.Stream(),torch.cuda.Stream()
 def peer():
  for send,recv in zip(sends,recvs):dist.all_to_all_single(recv,send,async_op=True).wait()
 peer();torch.cuda.synchronize()
 for n,recv in zip((8,16),recvs):
  for src in range(4):assert torch.all(recv[src*n:(src+1)*n]==src)
 if a.smoke:
  print(json.dumps(dict(status='PASS',mode=os.environ['MGO_TRANSPORT'],rank=rank,operation='all_to_all_single dispatch/combine',boot=boot,transport_env={k:v for k,v in os.environ.items() if k.startswith('NCCL_')})),flush=True)
 else:
  records=[]
  for case in ('h2d','peer','concurrent'):
   samples=[]
   for it in range(40):
    dist.barrier();torch.cuda.synchronize()
    start_h,end_h,start_p,end_p=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    if case!='peer':
     with torch.cuda.stream(hs):start_h.record();gpu.copy_(cpu,non_blocking=True);end_h.record()
    if case!='h2d':
     with torch.cuda.stream(ps):start_p.record();peer();end_p.record()
    torch.cuda.synchronize()
    if case!='peer':assert int(gpu[0])==17 and int(gpu[-1])==17
    if it>=10:samples.append(dict(h2d_ms=start_h.elapsed_time(end_h) if case!='peer' else 0.,peer_ms=start_p.elapsed_time(end_p) if case!='h2d' else 0.))
   records.append(dict(case=case,samples=samples,h2d_bytes=cpu.numel(),peer_tx_bytes=3*(8+16)*2048*2))
  out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
  (out/f'rank{rank}.json').write_text(json.dumps(dict(mode=os.environ['MGO_TRANSPORT'],rank=rank,boot=boot,records=records),indent=2)+'\n')
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
