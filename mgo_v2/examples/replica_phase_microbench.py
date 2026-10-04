#!/usr/bin/env python3
"""Model-free Env1/Env2 calibration for phase-aware MoE replication.

Env1: normal NCCL peer path (NVLink/NVSwitch when available).
Env2: NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1, matching the existing SHM control.

Measures on four ranks:
- 9-MiB pinned H2D, concurrent on all ranks;
- GPU-to-GPU activation one-way and round-trip for multiple row counts;
- 9-MiB expert D2D one-way, single pair and two concurrent pairs;
- resident-source D2D overlapping H2D;
- miss-source H2D(Ehot) -> D2D(Ehot) overlapping H2D(next);
- exact serial counterfactual H2D(Ehot) -> H2D(next) -> D2D(Ehot);
- synthetic Qwen expert compute for rows 64..2048.

No model checkpoint or offloading runtime is required.
"""
import argparse,json,os,time
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import()
import torch
import torch.distributed as dist
import torch.nn.functional as F

EB=9437184
ACT_ROWS=(1,8,16,32,64,128,256,512,1024,2048)
COMP_ROWS=(64,128,256,512,1024,2048)

def expert_kernel(x,gate,up,down):
 return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)

def stats(xs):
 if not xs:return {}
 y=sorted(float(x) for x in xs);n=len(y)
 def pct(p):return y[min(n-1,max(0,int(round((n-1)*p))))]
 return dict(median_ms=pct(.5),p90_ms=pct(.9),min_ms=y[0],max_ms=y[-1],samples_ms=list(xs))

def timed_wall(fn,rank,active=True):
 samples=[]
 for i in range(40):
  torch.cuda.synchronize();dist.barrier()
  t=time.perf_counter();fn();torch.cuda.synchronize()
  elapsed=(time.perf_counter()-t)*1e3
  # Keep the next iteration aligned, but never include this barrier in latency.
  dist.barrier()
  if active and i>=10:samples.append(elapsed)
 return stats(samples)

def p2p_oneway(send,recv,src,dst):
 if dist.get_rank()==src:dist.isend(send,dst=dst).wait()
 elif dist.get_rank()==dst:dist.irecv(recv,src=src).wait()

def p2p_roundtrip(send,recv,back,src,dst):
 rank=dist.get_rank()
 if rank==src:
  dist.isend(send,dst=dst).wait();dist.irecv(back,src=dst).wait()
 elif rank==dst:
  dist.irecv(recv,src=src).wait();dist.isend(recv,dst=src).wait()

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--environment',choices=['env1','env2'],required=True)
 a=p.parse_args()
 expected={'NCCL_CUMEM_ENABLE':'0'}
 if a.environment=='env2':expected.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
 actual={k:v for k,v in os.environ.items() if k.startswith('NCCL_')}
 for k,v in expected.items():assert actual.get(k)==v,(actual,expected)
 if a.environment=='env1':
  assert 'NCCL_P2P_DISABLE' not in actual and 'NCCL_IB_DISABLE' not in actual,(actual,expected)

 torch.set_num_threads(1);torch.cuda.set_device(0)
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 rank=dist.get_rank();world=dist.get_world_size();assert world==4
 a.output.mkdir(parents=True,exist_ok=True)

 host=[torch.empty(EB//2,dtype=torch.bfloat16,pin_memory=True).fill_(rank+1+i) for i in range(3)]
 dev=[torch.empty(EB//2,dtype=torch.bfloat16,device='cuda') for _ in range(3)]
 h2d_stream=torch.cuda.Stream();comm_stream=torch.cuda.Stream()
 first_done=torch.cuda.Event();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)

 gate=torch.empty((768,2048),dtype=torch.bfloat16,device='cuda').normal_(0,.01)
 up=torch.empty((768,2048),dtype=torch.bfloat16,device='cuda').normal_(0,.01)
 down=torch.empty((2048,768),dtype=torch.bfloat16,device='cuda').normal_(0,.01)

 out=dict(status='PASS',rank=rank,world=world,boot=BOOT,environment=a.environment,
          expert_bytes=EB,activation_row_bytes=4096,transport_env=actual,records=[])

 # 1) H2D all-rank concurrent.
 samples=[]
 for i in range(40):
  torch.cuda.synchronize();dist.barrier();begin.record()
  with torch.cuda.stream(h2d_stream):dev[0].copy_(host[0],non_blocking=True)
  h2d_stream.synchronize();end.record();end.synchronize()
  if i>=10:samples.append(begin.elapsed_time(end))
 out['records'].append(dict(kind='h2d_4rank',active=True,**stats(samples)))

 # 2) Activation G2G: one-way and round-trip, rank0<->rank1.
 for n in ACT_ROWS:
  send=torch.arange(n*2048,device='cuda',dtype=torch.int32).to(torch.bfloat16).reshape(n,2048)
  recv=torch.empty_like(send);back=torch.empty_like(send)
  s=timed_wall(lambda:p2p_oneway(send,recv,0,1),rank,rank in (0,1))
  out['records'].append(dict(kind='activation_oneway',rows=n,payload_bytes=n*4096,active=rank in (0,1),**s))
  s=timed_wall(lambda:p2p_roundtrip(send,recv,back,0,1),rank,rank in (0,1))
  out['records'].append(dict(kind='activation_roundtrip',rows=n,payload_bytes=n*4096,active=rank in (0,1),**s))
  del send,recv,back

 # 3) 9-MiB expert D2D, one pair.
 dev[0].fill_(rank+17)
 s=timed_wall(lambda:p2p_oneway(dev[0],dev[1],0,1),rank,rank in (0,1))
 out['records'].append(dict(kind='d2d_expert_onepair',payload_bytes=EB,active=rank in (0,1),**s))

 # 4) Two concurrent D2D pairs (0->1, 2->3).
 def two_pairs():
  if rank==0:dist.isend(dev[0],dst=1).wait()
  elif rank==1:dist.irecv(dev[1],src=0).wait()
  elif rank==2:dist.isend(dev[0],dst=3).wait()
  else:dist.irecv(dev[1],src=2).wait()
 s=timed_wall(two_pairs,rank,True)
 out['records'].append(dict(kind='d2d_expert_twopair',payload_bytes=EB,active=True,**s))

 # 5) Resident-source D2D || H2D.
 # rank0 copies resident dev0 to rank1 while all ranks perform one H2D.
 def resident_overlap():
  if rank==0:
   with torch.cuda.stream(h2d_stream):dev[2].copy_(host[2],non_blocking=True)
   work=dist.isend(dev[0],dst=1)
   work.wait();h2d_stream.synchronize()
  elif rank==1:
   with torch.cuda.stream(h2d_stream):dev[2].copy_(host[2],non_blocking=True)
   work=dist.irecv(dev[1],src=0);work.wait();h2d_stream.synchronize()
  else:
   with torch.cuda.stream(h2d_stream):dev[2].copy_(host[2],non_blocking=True)
   h2d_stream.synchronize()
 s=timed_wall(resident_overlap,rank,True)
 out['records'].append(dict(kind='resident_d2d_overlap_h2d',payload_bytes=EB,active=True,**s))

 # 6) Miss-source dependency: H2D(Ehot), then D2D(Ehot) overlaps H2D(Enext).
 def miss_overlap():
  if rank==0:
   with torch.cuda.stream(h2d_stream):
    dev[0].copy_(host[0],non_blocking=True);first_done.record(h2d_stream)
    dev[2].copy_(host[2],non_blocking=True)
   with torch.cuda.stream(comm_stream):
    comm_stream.wait_event(first_done)
   # NCCL host API cannot be launched from a CUDA stream; synchronize the
   # dependency only, then launch P2P while the second H2D may still be active.
   first_done.synchronize()
   work=dist.isend(dev[0],dst=1);work.wait()
   h2d_stream.synchronize()
  elif rank==1:
   work=dist.irecv(dev[1],src=0);work.wait()
  else:
   with torch.cuda.stream(h2d_stream):dev[2].copy_(host[2],non_blocking=True)
   h2d_stream.synchronize()
 s=timed_wall(miss_overlap,rank,True)
 out['records'].append(dict(kind='miss_h2d_then_d2d_overlap_next_h2d',payload_bytes=EB,active=True,**s))

 # 7) Exact serial counterfactual: H2D hot -> H2D next -> D2D.
 def miss_serial():
  if rank==0:
   with torch.cuda.stream(h2d_stream):
    dev[0].copy_(host[0],non_blocking=True)
    dev[2].copy_(host[2],non_blocking=True)
   h2d_stream.synchronize()
   dist.isend(dev[0],dst=1).wait()
  elif rank==1:
   dist.irecv(dev[1],src=0).wait()
  else:
   with torch.cuda.stream(h2d_stream):dev[2].copy_(host[2],non_blocking=True)
   h2d_stream.synchronize()
 s=timed_wall(miss_serial,rank,True)
 out['records'].append(dict(kind='miss_h2d_h2d_d2d_serial',payload_bytes=EB,active=True,**s))

 # 8) Expert compute CUDA-event timing.
 for n in COMP_ROWS:
  x=torch.empty((n,2048),dtype=torch.bfloat16,device='cuda').normal_(0,.01)
  samples=[]
  for i in range(40):
   torch.cuda.synchronize();begin.record()
   y=expert_kernel(x,gate,up,down)
   end.record();end.synchronize()
   if i>=10:samples.append(begin.elapsed_time(end))
  assert y.shape==(n,2048)
  out['records'].append(dict(kind='expert_compute',rows=n,active=True,**stats(samples)))
  del x,y

 (a.output/f'rank{rank}.json').write_text(json.dumps(out,indent=2)+'\n')
 dist.barrier()
 if rank==0:
  meta=dict(status='PASS',environment=a.environment,world=world,
            physical_visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
            transport_env=actual,
            note='wall-time G2G/overlap cases include host/NCCL launch and barriers; expert compute/H2D use CUDA timing where noted')
  (a.output/'status.json').write_text(json.dumps(meta,indent=2)+'\n')
 dist.destroy_process_group()

if __name__=='__main__':main()
