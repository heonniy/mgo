"""Fast R4 expert-fetch microbenchmark: current staging vs direct pinned source.

Physical GPUs are fixed by the runner to 0,1,4,5.  This does not run the model.
It maps the existing CPU expert store, prefaults only the selected experts, and
compares:
  A) current runtime path: pageable/file-backed expert -> 2-slot pinned stage -> H2D
  B) direct-pinned upper bound: prebuilt pinned expert pool -> H2D

Measured conditions:
  isolated rank 0/1/2/3, pairs (0,1)/(2,3), all four ranks.
Burst sizes: 1, 8, 16, 32 experts.  Five measured repeats after one warmup.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import()

import argparse,json,statistics,struct,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from mgo_v2.pinned_h2d import PriorityH2DScheduler,copy_expert_to_stage

GPUS=[0,1,4,5]
STORE=Path('/home/hwlee/mgo-results/runtime_validation_20261001/expert_store')
EB=9437184
MAX_BURST=32
BURSTS=(1,8,16,32)
REPEATS=5
WARMUPS=1
# Spread keys over the model so the test is not one contiguous file region.
KEYS=[(17+i*173)%(48*128) for i in range(MAX_BURST)]
CONDITIONS=[
 ('iso0',(0,)),('iso1',(1,)),('iso2',(2,)),('iso3',(3,)),
 ('pair01',(0,1)),('pair23',(2,3)),('all4',(0,1,2,3)),
]

def write(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)

def stats(xs):
 xs=[float(x) for x in xs]
 if not xs:return dict(n=0)
 mean=statistics.mean(xs);sd=statistics.pstdev(xs) if len(xs)>1 else 0.0
 return dict(n=len(xs),median=statistics.median(xs),mean=mean,min=min(xs),max=max(xs),cv=(sd/mean if mean else 0.0))

def load_selected_experts():
 param=STORE/'archer_param_0';index_path=STORE/'archer_index'
 assert param.exists() and index_path.exists(),STORE
 backing=torch.from_file(str(param),shared=False,size=param.stat().st_size,dtype=torch.uint8)
 index={}
 with index_path.open('rb') as f:
  count=struct.unpack('<I',f.read(4))[0]
  for _ in range(count):
   tid,file_id,offset,size,ndim=struct.unpack('<IIqQq',f.read(32))
   shape=struct.unpack('<'+'q'*ndim,f.read(8*ndim));f.read(6)
   assert file_id==0;index[tid]=(offset,size,shape)
 experts=[]
 for key in KEYS:
  tensors=[]
  for i in range(3):
   offset,size,shape=index[key*3+i]
   t=backing[offset:offset+size].view(torch.bfloat16).view(shape)
   tensors.append(t)
  assert sum(t.numel()*t.element_size() for t in tensors)==EB
  # Prefault selected source pages before any measurement.
  _=sum(float(t.reshape(-1)[::4096].float().sum()) for t in tensors)
  experts.append(tuple(tensors))
 return backing,experts

def build_direct_pinned(experts):
 pool=torch.empty((MAX_BURST,EB//2),dtype=torch.bfloat16,device='cpu',pin_memory=True)
 assert pool.is_pinned()
 for i,tensors in enumerate(experts):copy_expert_to_stage(pool[i],tensors,'torch')
 return pool

def current_once(cache,experts,n,active,rank):
 if not active:
  dist.barrier();dist.barrier();return None
 sched=PriorityH2DScheduler(cache,stage_count=2,profile=True,staging_backend='torch')
 dist.barrier();start=time.perf_counter()
 for i in range(n):sched.enqueue_demand(i,KEYS[i],experts[i])
 sched.synchronize();torch.cuda.synchronize()
 wall_ms=(time.perf_counter()-start)*1000
 dist.barrier()
 traces=list(sched.trace);assert len(traces)==n
 dma_ms=[];stage_ms=[];queue_ms=[]
 for t in traces:
  t.done.synchronize()
  dma_ms.append(float(t.begin.elapsed_time(t.done)))
  stage_ms.append((t.staging_finished_at-t.staging_started_at)*1000)
  queue_ms.append((t.staging_started_at-t.queued_at)*1000)
 sched.close()
 return dict(wall_ms=wall_ms,dma_ms=dma_ms,stage_ms=stage_ms,queue_ms=queue_ms)

def direct_once(cache,pool,n,active,rank):
 stream=torch.cuda.Stream()
 begins=[torch.cuda.Event(enable_timing=True) for _ in range(n)]
 ends=[torch.cuda.Event(enable_timing=True) for _ in range(n)]
 if not active:
  dist.barrier();dist.barrier();return None
 dist.barrier();start=time.perf_counter()
 with torch.cuda.stream(stream):
  for i in range(n):
   begins[i].record(stream);cache[i].copy_(pool[i],non_blocking=True);ends[i].record(stream)
 stream.synchronize();torch.cuda.synchronize()
 wall_ms=(time.perf_counter()-start)*1000
 dist.barrier()
 dma_ms=[float(b.elapsed_time(e)) for b,e in zip(begins,ends)]
 return dict(wall_ms=wall_ms,dma_ms=dma_ms,stage_ms=[],queue_ms=[])

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
 physical=[int(x) for x in os.environ['MGO_V2_PHYSICAL_GPUS'].split(',')];assert physical==GPUS
 assert int(os.environ['CUDA_VISIBLE_DEVICES'])==GPUS[rank],(rank,os.environ['CUDA_VISIBLE_DEVICES'])
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.20)
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 # Match runtime CPU placement if topology receipt exists.
 topo=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json')
 cpus=None
 if topo.exists():
  cpus=json.loads(topo.read_text())['fixed_affinity'][str(GPUS[rank])]
  for task in Path('/proc/self/task').iterdir():
   try:os.sched_setaffinity(int(task.name),cpus)
   except FileNotFoundError:pass
 backing,experts=load_selected_experts()
 direct_pool=build_direct_pinned(experts)
 cache=torch.empty((MAX_BURST,EB//2),dtype=torch.bfloat16,device='cuda')
 # Touch all direct-pinned pages before timing.
 _=float(direct_pool[:,::4096].float().sum())
 rows=[]
 for condition,active_ranks in CONDITIONS:
  active=rank in active_ranks
  for n in BURSTS:
   for mode in ('current_staging','direct_pinned'):
    fn=current_once if mode=='current_staging' else direct_once
    for rep in range(WARMUPS+REPEATS):
     row=fn(cache,experts if mode=='current_staging' else direct_pool,n,active,rank)
     if rep>=WARMUPS and active:
      rows.append(dict(condition=condition,active_ranks=list(active_ranks),burst=n,mode=mode,
       repeat=rep-WARMUPS,rank=rank,physical_gpu=GPUS[rank],**row))
     dist.barrier()
 write(a.output/f'rank{rank}.json',dict(status='PASS',rank=rank,physical_gpu=GPUS[rank],
  expert_bytes=EB,keys=KEYS,bursts=list(BURSTS),repeats=REPEATS,pinned_pool_bytes=direct_pool.numel()*direct_pool.element_size(),
  cpu_affinity=cpus,bootstrap=BOOT,rows=rows))
 dist.barrier()
 if rank==0:
  all_rows=[]
  for r in range(4):all_rows.extend(json.loads((a.output/f'rank{r}.json').read_text())['rows'])
  summary=[]
  for condition,_active in CONDITIONS:
   for n in BURSTS:
    selected=[x for x in all_rows if x['condition']==condition and x['burst']==n]
    by_mode={}
    for mode in ('current_staging','direct_pinned'):
     m=[x for x in selected if x['mode']==mode]
     by_rank={}
     for rr in sorted({x['rank'] for x in m}):
      q=[x for x in m if x['rank']==rr]
      by_rank[str(rr)]=dict(
       physical_gpu=GPUS[rr],
       wall_ms=stats([x['wall_ms'] for x in q]),
       dma_ms=stats([v for x in q for v in x['dma_ms']]),
       stage_ms=stats([v for x in q for v in x['stage_ms']]),
       queue_ms=stats([v for x in q for v in x['queue_ms']]))
     # Strict-runtime relevant metric: slowest active rank per repeat.
     critical=[]
     for rep in range(REPEATS):
      vals=[x['wall_ms'] for x in m if x['repeat']==rep]
      if vals:critical.append(max(vals))
     by_mode[mode]=dict(by_rank=by_rank,critical_rank_wall_ms=stats(critical))
    cur=by_mode['current_staging']['critical_rank_wall_ms'].get('median')
    pin=by_mode['direct_pinned']['critical_rank_wall_ms'].get('median')
    summary.append(dict(condition=condition,burst=n,modes=by_mode,
      direct_pinned_gain=(1-pin/cur if cur and pin else None)))
  write(a.output/'summary.json',dict(status='PASS',
   question='rank-specific simultaneous-fetch degradation and direct-pinned staging-removal headroom',
   physical_gpus=GPUS,expert_bytes=EB,bursts=list(BURSTS),repeats=REPEATS,summary=summary))
 dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
