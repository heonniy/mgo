"""Fixed-size H2D concurrency / empirical SHM pairs; no physical NUMA labels."""
import os,json,argparse,time,itertools
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
boot=pin_rank_before_cuda_import()
root=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004')
topo=json.loads((root/'topology.json').read_text());gpu=int(boot['visible_gpu']);cpus=topo['fixed_affinity'][str(gpu)];os.sched_setaffinity(0,cpus)
import torch
import torch.distributed as dist
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--kind',choices=['h2d','shm'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.cuda.set_device(0)
dist.init_process_group('gloo' if a.kind=='h2d' else 'nccl',**({} if a.kind=='h2d' else {'device_id':torch.device('cuda:0')}))
rank=dist.get_rank();world=dist.get_world_size();rows=[]
if a.kind=='h2d':
 host=torch.full((9*1024**2,),31,dtype=torch.uint8,pin_memory=True);target=torch.empty_like(host,device='cuda');assert host.is_pinned()
 times=[];wall=[];aggregate=[];skews=[]
 begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 for i in range(70):
  dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
  begin.record();target.copy_(host,non_blocking=True);end.record();end.synchronize();finish=time.perf_counter()
  duration=begin.elapsed_time(end);all_times=[None]*world;dist.all_gather_object(all_times,(start,finish))
  if i>=20:
   times.append(duration);wall.append(finish-start);span=max(x[1] for x in all_times)-min(x[0] for x in all_times)
   aggregate.append(world*host.numel()/span/1e9);skews.append(max(x[0] for x in all_times)-min(x[0] for x in all_times))
 assert torch.equal(target.cpu(),host)
 bw=[host.numel()/(ms/1000)/1e9 for ms in times]
 rows.append(dict(kind='H2D',gpu=gpu,concurrency=world,payload_bytes=host.numel(),warmups=20,repeats=50,median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),median_GBps=float(np.median(bw)),p90_GBps=float(np.percentile(bw,90)),aggregate_wall_median_GBps=float(np.median(aggregate)),aggregate_wall_p90_GBps=float(np.percentile(aggregate,90)),samples_ms=times,samples_GBps=bw,aggregate_wall_samples_GBps=aggregate,start_skew_seconds=skews,affinity=cpus,boot=boot))
else:
 assert world==8
 cpu=dist.new_group(backend='gloo')
 for low,high in itertools.combinations(range(8),2):
  for kib in [32,128,512]:
   dist.barrier(group=cpu)
   if rank in (low,high):
    send=torch.full((kib*1024,),47,dtype=torch.uint8,device='cuda');recv=torch.empty_like(send);times=[]
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);torch.cuda.synchronize()
    for i in range(1 if a.preflight else 70):
     begin.record()
     if rank==low:dist.send(send,high);dist.recv(recv,high)
     else:dist.recv(recv,low);dist.send(recv,low)
     end.record();end.synchronize()
     if i>=20:times.append(begin.elapsed_time(end))
    assert torch.equal(recv,send)
    if a.preflight:
     rows.append(dict(kind='SHM_transport_preflight',pair=[low,high],payload_bytes=kib*1024,payload_valid=True))
    else:
     rows.append(dict(kind='SHM_dispatch_return',gpu=gpu,pair=[low,high],initiator=rank==low,payload_bytes=kib*1024,warmups=20,repeats=50,median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),samples_ms=times,affinity=cpus,boot=boot))
   dist.barrier(group=cpu)
 dist.destroy_process_group(cpu)
dist.barrier();dist.destroy_process_group()
(a.output/f'rank{rank}.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n')
