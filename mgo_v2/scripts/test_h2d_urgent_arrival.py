"""Urgent arrival after background starts: ordering and bounded queue latency."""
import json,time,statistics
from pathlib import Path
import torch
from mgo_v2.pinned_h2d import PriorityH2DScheduler

def main():
 torch.set_num_threads(2);torch.cuda.set_device(0)
 n=9437184//2;cache=torch.empty((16,n),device='cuda',dtype=torch.bfloat16);host=torch.ones(n,dtype=torch.bfloat16)
 base=[];busy=[];order_checks=0
 for repeat in range(15):
  q=PriorityH2DScheduler(cache,profile=True);t=q.enqueue_demand(15,15,[host]);q._submitted(t);base.append((t.submitted_at-t.queued_at)*1000);q.close()
  q=PriorityH2DScheduler(cache,profile=True)
  for i in range(8):q.enqueue_prefetch(i,i,[host])
  q._submitted(q.tickets[0]);t=q.enqueue_demand(15,15,[host]);q._submitted(t);busy.append((t.submitted_at-t.queued_at)*1000);q.close()
  assert all(x.submitted_at>t.submitted_at for x in q.trace if not x.state.urgent and x.submitted_at>t.queued_at)
  assert bool((cache[15]==1).all());order_checks+=1
 # An in-progress CPU stage or DMA is not preempted. Allow 1ms noise + one
 # observed copy duration, while still requiring strict not-yet-started ordering.
 dma=[x.begin.elapsed_time(x.done) for x in q.trace]
 bound=max(dma)+1.0;delta=statistics.median(busy[3:])-statistics.median(base[3:]);assert delta<=bound,(delta,bound)
 result=dict(status='PASS',arrival_order_checks=order_checks,baseline_median_ms=statistics.median(base[3:]),background_active_median_ms=statistics.median(busy[3:]),extra_median_ms=delta,allowed_extra_ms=bound,bound='one observed copy duration + 1ms host scheduling allowance; strict queued-order check separately',samples=dict(baseline=base,background_active=busy))
 p=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004/M8_urgent_arrival.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
