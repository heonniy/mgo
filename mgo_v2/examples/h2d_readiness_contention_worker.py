"""R4 H2D readiness/contention characterization on the production expert-copy path.

Measures per-rank queueing, CPU pageable->pinned staging, H2D DMA, and completion
latency under isolated, paired, all-rank, and all-rank+A2A conditions.
Only the caller-selected CUDA_VISIBLE_DEVICES are visible to workers.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import()

import argparse,ctypes,json,statistics,struct,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from mgo_v2.model_loader import prepare_checkpoint_store
from mgo_v2.pinned_h2d import PriorityH2DScheduler

MODEL='/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
STORE=Path('/home/hwlee/mgo-results/runtime_validation_20261001/expert_store')
EB=9437184
PHYSICAL=[0,1,4,5]
COPIES=32
REPEATS=8
A2A_BYTES=64*2**20

def write(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)

def percentile(xs,p):
 if not xs:return None
 return float(np.percentile(np.asarray(xs,np.float64),p))

def stats(xs):
 xs=[float(x) for x in xs]
 if not xs:return dict(n=0)
 mean=float(statistics.mean(xs));sd=float(statistics.pstdev(xs)) if len(xs)>1 else 0.0
 return dict(n=len(xs),mean=mean,median=float(statistics.median(xs)),p95=percentile(xs,95),minimum=min(xs),maximum=max(xs),cv=(sd/mean if mean else 0.0))

def load_experts():
 manifest=prepare_checkpoint_store(MODEL,STORE)
 backing=torch.from_file(str(STORE/'archer_param_0'),shared=False,size=manifest['bytes'],dtype=torch.uint8)
 # Match the inference path: fault the full read-only expert pool into host RAM
 # before measurement so this experiment is RAM/NUMA/PCIe, not disk I/O.
 _=backing[::4096].to(torch.int64).sum().item()
 index={}
 with (STORE/'archer_index').open('rb') as f:
  count=struct.unpack('<I',f.read(4))[0]
  for _ in range(count):
   tid,file_id,offset,size,ndim=struct.unpack('<IIqQq',f.read(32));shape=struct.unpack('<'+'q'*ndim,f.read(8*ndim));f.read(6)
   assert file_id==0;index[tid]=(offset,size,shape)
 experts=[]
 for key in range(48*128):
  tensors=[]
  for i in range(3):
   offset,size,shape=index[key*3+i]
   tensors.append(backing[offset:offset+size].view(torch.bfloat16).view(shape))
  assert sum(t.numel()*2 for t in tensors)==EB
  experts.append(tuple(tensors))
 return backing,experts

def condition_table():
 return [
  ('iso_r0',(0,),False),('iso_r1',(1,),False),('iso_r2',(2,),False),('iso_r3',(3,),False),
  ('pair_r01',(0,1),False),('pair_r23',(2,3),False),
  ('all4',(0,1,2,3),False),
  ('all4_a2a',(0,1,2,3),True),
 ]

def run_once(rank,condition,active,with_a2a,repeat,cache,experts,cpu_team):
 active_here=rank in active
 scheduler=PriorityH2DScheduler(cache,stage_count=2,profile=True,staging_backend='torch',cpu_team=cpu_team)
 # A common GPU timestamp lets us observe when each expert becomes ready after
 # the synchronized condition start, including host-side queue/staging delay.
 origin=torch.cuda.Event(enable_timing=True)
 with torch.cuda.stream(scheduler.h2d_stream):origin.record()
 origin.synchronize()
 dist.barrier();torch.cuda.synchronize()
 start=time.perf_counter()
 if active_here:
  base=(rank*1536+repeat*211)%len(experts)
  for i in range(COPIES):
   key=(base+i*37)%len(experts)
   scheduler.enqueue_demand(i,key,experts[key])
 comm_ms=None
 if with_a2a:
  assert tuple(active)==(0,1,2,3)
  elems=A2A_BYTES//2
  assert elems%4==0
  send=torch.full((elems,),float(rank),dtype=torch.bfloat16,device='cuda')
  recv=torch.empty_like(send)
  cs,ce=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  cs.record();work=dist.all_to_all_single(recv,send,async_op=True);ce.record()
  # Production-like overlap: H2D runs on its dedicated stream while NCCL is in flight.
  if active_here:scheduler.synchronize()
  work.wait();torch.cuda.synchronize();comm_ms=float(cs.elapsed_time(ce))
  del send,recv
 elif active_here:
  scheduler.synchronize();torch.cuda.synchronize()
 end=time.perf_counter()
 tickets=[]
 if active_here:
  assert len(scheduler.trace)==COPIES,(condition,rank,len(scheduler.trace))
  for t in scheduler.trace:
   t.done.synchronize()
   assert t.staging_started_at is not None and t.staging_finished_at is not None and t.submitted_at is not None
   queue_ms=(t.staging_started_at-t.queued_at)*1000
   staging_ms=(t.staging_finished_at-t.staging_started_at)*1000
   dma_ms=float(t.begin.elapsed_time(t.done))
   ready_ms=float(origin.elapsed_time(t.done))
   tickets.append(dict(slot=t.slot,key=t.key,queue_ms=queue_ms,staging_ms=staging_ms,dma_ms=dma_ms,ready_ms_from_origin=ready_ms,
                       dma_GBps=(scheduler.bytes_per_expert/(dma_ms*1e6) if dma_ms>0 else None)))
 row=dict(condition=condition,repeat=repeat,rank=rank,physical_gpu=PHYSICAL[rank],active=active_here,active_ranks=list(active),
          with_a2a=with_a2a,copies=COPIES if active_here else 0,bytes=(COPIES*EB if active_here else 0),
          batch_wall_ms=(end-start)*1000 if active_here else None,comm_ms=comm_ms,tickets=tickets,
          scheduler_metrics=dict(scheduler.metrics),cpu_team=cpu_team,bootstrap=BOOT)
 scheduler.close();dist.barrier()
 return row

def summarize(rows):
 active=[r for r in rows if r['active']]
 by_condition={}
 for cond in sorted({r['condition'] for r in rows}):
  cr=[r for r in active if r['condition']==cond]
  by_rank={}
  for rank in sorted({r['rank'] for r in cr}):
   rr=[r for r in cr if r['rank']==rank];tickets=[t for r in rr for t in r['tickets']]
   by_rank[str(rank)]=dict(
    physical_gpu=PHYSICAL[rank],
    batch_wall_ms=stats([r['batch_wall_ms'] for r in rr]),
    queue_ms=stats([t['queue_ms'] for t in tickets]),
    staging_ms=stats([t['staging_ms'] for t in tickets]),
    dma_ms=stats([t['dma_ms'] for t in tickets]),
    ready_ms_from_origin=stats([t['ready_ms_from_origin'] for t in tickets]),
    dma_GBps=stats([t['dma_GBps'] for t in tickets]),
    comm_ms=stats([r['comm_ms'] for r in rr if r['comm_ms'] is not None]),
   )
  by_condition[cond]=by_rank
 return by_condition

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
 physical=int(os.environ.get('MGO_V2_PHYSICAL_GPUS','0,1,4,5').split(',')[rank]);assert physical==PHYSICAL[rank]
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.25)
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 topology=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json')
 cpus=json.loads(topology.read_text())['fixed_affinity'][str(physical)]
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 backing,experts=load_experts()
 cache=torch.empty((COPIES,EB//2),dtype=torch.bfloat16,device='cuda')
 rows=[]
 for condition,active,with_a2a in condition_table():
  for repeat in range(REPEATS):
   rows.append(run_once(rank,condition,active,with_a2a,repeat,cache,experts,cpus[1:3]))
 write(a.output/f'rank{rank}.json',dict(status='PASS',rank=rank,physical_gpu=physical,expert_bytes=EB,copies_per_active_rank=COPIES,repeats=REPEATS,a2a_bytes_per_rank=A2A_BYTES,rows=rows))
 dist.barrier()
 if rank==0:
  all_rows=[]
  for r in range(world):all_rows.extend(json.loads((a.output/f'rank{r}.json').read_text())['rows'])
  write(a.output/'summary.json',dict(status='PASS',scope='R4 H2D readiness variability: isolated, paired, all4, all4+A2A',physical_gpus=PHYSICAL,
       expert_bytes=EB,copies_per_active_rank=COPIES,repeats=REPEATS,a2a_bytes_per_rank=A2A_BYTES,conditions=summarize(all_rows)))
 dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
