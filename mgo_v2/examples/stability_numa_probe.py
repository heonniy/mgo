"""Bounded local-domain probes; never fabricate an unavailable remote node."""
import os,json,argparse,time,ctypes
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
boot=pin_rank_before_cuda_import()
root=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004')
topo=json.loads((root/'topology.json').read_text());gpu=int(boot['visible_gpu']);cpus=topo['fixed_affinity'][str(gpu)];os.sched_setaffinity(0,cpus)
import torch
import torch.distributed as dist
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['h2d','shm'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.cuda.set_device(0)
rows=[]
if a.kind=='h2d':
 host=torch.full((9*1024**2,),17,dtype=torch.uint8,pin_memory=True);target=torch.empty_like(host,device='cuda');assert host.is_pinned()
 lib=ctypes.CDLL('libnuma.so.1',use_errno=True);node=ctypes.c_int(-1)
 lib.get_mempolicy.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong]
 rc=lib.get_mempolicy(ctypes.byref(node),None,0,ctypes.c_void_p(host.data_ptr()),3);assert rc==0 and node.value==boot['numa_node']
 times=[]
 for i in range(70):
  begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  begin.record();target.copy_(host,non_blocking=True);end.record();end.synchronize()
  if i>=20:times.append(begin.elapsed_time(end))
 assert torch.equal(target.cpu(),host)
 rows.append(dict(kind='H2D',gpu=gpu,locality='local_os_node',node=node.value,payload_bytes=host.numel(),warmups=20,repeats=50,median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),samples_ms=times,affinity=cpus,boot=boot))
 rank=0
else:
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'));rank=dist.get_rank();assert dist.get_world_size()==2
 for kib in [32,128,512]:
  send=torch.full((kib*1024,),23,dtype=torch.uint8,device='cuda');recv=torch.empty_like(send);times=[]
  dist.barrier();torch.cuda.synchronize()
  for i in range(70):
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
   if rank==0:dist.send(send,1);dist.recv(recv,1)
   else:dist.recv(recv,0);dist.send(recv,0)
   end.record();end.synchronize()
   if i>=20:times.append(begin.elapsed_time(end))
  assert torch.equal(recv,send)
  rows.append(dict(kind='SHM_dispatch_return',gpu=gpu,locality='same_os_node',pair=[0,1],payload_bytes=kib*1024,warmups=20,repeats=50,median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),samples_ms=times,affinity=cpus,boot=boot))
 dist.barrier();dist.destroy_process_group()
(a.output/f'rank{rank}.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n')
