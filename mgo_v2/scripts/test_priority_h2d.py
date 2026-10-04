"""Physical worker scheduling, data hazards, cancellation and enqueue latency."""
import json,time,statistics
from pathlib import Path
import torch
from mgo_v2.pinned_h2d import PriorityH2DScheduler,PinnedH2DCache

def main():
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.25)
 n=9437184//2;cache=torch.empty((16,n),dtype=torch.bfloat16,device='cuda');backing=[torch.full((n,),float(i),dtype=torch.bfloat16) for i in range(16)]
 q=PriorityH2DScheduler(cache,profile=True,autostart=False)
 bg=q.enqueue_prefetch(0,0,[backing[0]]);q.enqueue_prefetch(1,1,[backing[1]])
 urgent=q.enqueue_demand(2,2,[backing[2]]);promoted=q.enqueue_prefetch(3,3,[backing[3]])
 assert q.promote(3,3) is promoted and q.enqueue_demand(3,3,[backing[3]]) is promoted
 canceled=q.enqueue_prefetch(4,4,[backing[4]]);assert q.cancel_if_queued(4,4)
 q.start();q.synchronize();assert q.trace[0].key==2 and q.trace[1].key==3
 assert q.metrics['copies']==4 and sum(t.key==3 for t in q.trace)==1
 for slot in range(4):assert bool((cache[slot]==slot).all())
 # An old compute read must finish before worker overwrites the same slot.
 q.wait_for_slot(1);before=cache[1].clone();q.record_slot_use(1)
 q.enqueue_demand(1,5,[backing[5]]);q.wait_for_slot(1);after=cache[1].clone();torch.cuda.synchronize()
 assert bool((before==1).all()) and bool((after==5).all())
 q.close()
 # Many stage reuses must never overwrite a DMA source still in flight.
 q=PriorityH2DScheduler(cache,profile=True)
 for i in range(16):q.enqueue_demand(i,i,[backing[i]])
 q.synchronize()
 for i in range(16):assert bool((cache[i]==i).all())
 q.close()
 # Submission latency is not DMA latency; compare identical expert batches.
 legacy=[];worker=[];old=PinnedH2DCache(cache,2)
 for repeat in range(12):
  torch.cuda.synchronize();start=time.perf_counter()
  for i in range(8):old.enqueue(i,[backing[i]])
  legacy.append((time.perf_counter()-start)*1000);old.synchronize()
  q=PriorityH2DScheduler(cache);start=time.perf_counter()
  for i in range(8):q.enqueue_demand(i,i,[backing[i]])
  worker.append((time.perf_counter()-start)*1000);q.close()
 assert statistics.median(worker)<statistics.median(legacy),(worker,legacy)
 # Actual CUDA copy intervals overlap a default-stream GEMM window.
 q=PriorityH2DScheduler(cache,profile=True);x=torch.randn((4096,4096),device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
 torch.mm(x,x,out=y);torch.cuda.synchronize()
 for i in range(16):q.enqueue_demand(i,i,[backing[i]])
 begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
 for _ in range(32):torch.mm(x,x,out=y)
 end.record();q.synchronize();torch.cuda.synchronize()
 window=(q.trace_origin.elapsed_time(begin),q.trace_origin.elapsed_time(end));copies=[(q.trace_origin.elapsed_time(t.begin),q.trace_origin.elapsed_time(t.done)) for t in q.trace]
 from mgo_v2.phase_timing import intersection_ms,union_ms
 overlap=intersection_ms(copies,[window]);assert overlap>0
 q.close()
 result=dict(status='PASS',urgent_overtakes_queued_background=True,promoted_copy_not_duplicated=True,queued_cancel=True,prior_compute_protected=True,stage_reuse_content_checks=16,enqueue_batch8_ms=dict(legacy_median=statistics.median(legacy),worker_median=statistics.median(worker)),DMA_total_ms=union_ms(copies),DMA_overlap_GEMM_window_ms=overlap,profile_note='Copy-stream CUDA intervals vs default-stream GEMM window; final kernel-only exposure uses Nsight',unit='9MiB experts')
 p=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004/M8_scheduler_microbench.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
