#!/usr/bin/env python3
"""Model-free calibration for replica phase scheduling on four GPUs.

Measures:
- pinned 9-MiB H2D;
- 9-MiB GPU->GPU replica copy via NCCL P2P;
- miss-source pipeline: H2D(Ehot), then D2D(Ehot) overlapping H2D(Enext);
- synthetic Qwen expert compute for row counts up to 2048.

No model checkpoint or expert offloading runtime is required.
"""
import argparse,json,os,time
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import()
import torch
import torch.distributed as dist
import torch.nn.functional as F

EB=9437184
ROWS=(64,128,256,512,1024,2048)

def expert_kernel(x,gate,up,down):
 return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)

def stats(xs):
 y=sorted(xs);n=len(y)
 def pct(p):return y[min(n-1,max(0,int(round((n-1)*p))))]
 return dict(median_ms=pct(.5),p90_ms=pct(.9),samples_ms=xs)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.cuda.set_device(0)
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 rank=dist.get_rank();world=dist.get_world_size();assert world==4
 a.output.mkdir(parents=True,exist_ok=True)

 host=[torch.empty(EB//2,dtype=torch.bfloat16,pin_memory=True).fill_(rank+1+i) for i in range(2)]
 dev=[torch.empty(EB//2,dtype=torch.bfloat16,device='cuda') for _ in range(2)]
 h2d_stream=torch.cuda.Stream();comm_stream=torch.cuda.Stream()
 first_done=torch.cuda.Event();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)

 # Representative expert weights for compute timing.
 gate=torch.empty((768,2048),dtype=torch.bfloat16,device='cuda').normal_(0,.01)
 up=torch.empty((768,2048),dtype=torch.bfloat16,device='cuda').normal_(0,.01)
 down=torch.empty((2048,768),dtype=torch.bfloat16,device='cuda').normal_(0,.01)

 out=dict(status='PASS',rank=rank,world=world,boot=BOOT,expert_bytes=EB,records=[])

 # H2D, all four ranks concurrent.
 samples=[]
 for i in range(40):
  torch.cuda.synchronize();dist.barrier();begin.record()
  with torch.cuda.stream(h2d_stream):dev[0].copy_(host[0],non_blocking=True)
  h2d_stream.synchronize();end.record();end.synchronize()
  if i>=10:samples.append(begin.elapsed_time(end))
 out['records'].append(dict(kind='h2d_4rank',**stats(samples)))

 # D2D P2P, rank0 -> rank1, other ranks synchronize only.
 samples=[]
 for i in range(40):
  dev[0].fill_(rank+17);torch.cuda.synchronize();dist.barrier()
  start=time.perf_counter()
  if rank==0:
   work=dist.isend(dev[0],dst=1);work.wait()
  elif rank==1:
   work=dist.irecv(dev[1],src=0);work.wait()
  torch.cuda.synchronize();dist.barrier()
  elapsed=(time.perf_counter()-start)*1e3
  if rank in (0,1) and i>=10:samples.append(elapsed)
 out['records'].append(dict(kind='d2d_p2p_9mib',active=rank in (0,1),**stats(samples) if samples else {}))

 # Miss-source overlap:
 # rank0 H2D Ehot; as soon as it completes, send Ehot to rank1 while rank0
 # starts H2D Enext. Other ranks participate only in boundaries.
 samples=[]
 for i in range(40):
  torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
  if rank==0:
   with torch.cuda.stream(h2d_stream):
    dev[0].copy_(host[0],non_blocking=True);first_done.record(h2d_stream)
    dev[1].copy_(host[1],non_blocking=True)
   with torch.cuda.stream(comm_stream):
    comm_stream.wait_event(first_done)
    work=dist.isend(dev[0],dst=1)
   work.wait();h2d_stream.synchronize();comm_stream.synchronize()
  elif rank==1:
   with torch.cuda.stream(comm_stream):
    work=dist.irecv(dev[1],src=0)
   work.wait();comm_stream.synchronize()
  torch.cuda.synchronize();dist.barrier()
  elapsed=(time.perf_counter()-start)*1e3
  if rank in (0,1) and i>=10:samples.append(elapsed)
 out['records'].append(dict(kind='h2d_then_d2d_overlap_h2d',active=rank in (0,1),**stats(samples) if samples else {}))

 # Counterfactual serial H2D(Ehot), H2D(Enext), then D2D(Ehot).
 samples=[]
 for i in range(40):
  torch.cuda.synchronize();dist.barrier();start=time.perf_counter()
  if rank==0:
   with torch.cuda.stream(h2d_stream):
    dev[0].copy_(host[0],non_blocking=True);dev[1].copy_(host[1],non_blocking=True)
   h2d_stream.synchronize()
   work=dist.isend(dev[0],dst=1);work.wait()
  elif rank==1:
   work=dist.irecv(dev[1],src=0);work.wait()
  torch.cuda.synchronize();dist.barrier()
  elapsed=(time.perf_counter()-start)*1e3
  if rank in (0,1) and i>=10:samples.append(elapsed)
 out['records'].append(dict(kind='h2d_h2d_d2d_serial',active=rank in (0,1),**stats(samples) if samples else {}))

 # Expert compute. Each rank times independently; no communication.
 for n in ROWS:
  x=torch.empty((n,2048),dtype=torch.bfloat16,device='cuda').normal_(0,.01)
  samples=[]
  for i in range(40):
   torch.cuda.synchronize();begin.record()
   y=expert_kernel(x,gate,up,down)
   end.record();end.synchronize()
   if i>=10:samples.append(begin.elapsed_time(end))
  assert y.shape==(n,2048)
  out['records'].append(dict(kind='expert_compute',rows=n,**stats(samples)))
  del x,y

 (a.output/f'rank{rank}.json').write_text(json.dumps(out,indent=2)+'\n')
 dist.barrier()
 if rank==0:
  meta=dict(status='PASS',world=world,physical_visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
            note='wall-time overlap cases include NCCL host launch/barrier overhead; expert compute uses CUDA events')
  (a.output/'status.json').write_text(json.dumps(meta,indent=2)+'\n')
 dist.destroy_process_group()

if __name__=='__main__':main()
