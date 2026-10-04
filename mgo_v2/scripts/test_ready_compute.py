"""Ordering invariants and real-copy/GEMM exposure diagnostic."""
import json,statistics
from pathlib import Path
from mgo_v2.ready_compute import expert_order

def unit():
 class Fake:
  def __init__(self):self.available={2,3};self.waits=[]
  def ready(self,s):return s in self.available
  def wait_for_slot(self,s):
   assert not any(x in self.available for x in range(4) if x not in emitted)
   self.waits.append(s)
 f=Fake();emitted=[]
 for i in expert_order([(0,),(1,),(2,),(3,)],f,True):emitted.append(i)
 assert emitted==[2,3,0,1] and f.waits==[0,1]
 print('PASS ready traversal ordering')

def gpu():
 import torch
 from mgo_v2.pinned_h2d import PriorityH2DScheduler
 from mgo_v2.phase_timing import union_ms,intersection_ms
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
 n=9437184//2;cache=torch.empty((12,n),device='cuda',dtype=torch.bfloat16);x=torch.randn((128,2048),device='cuda',dtype=torch.bfloat16)
 host=[torch.full((n,),float(i+1)/128,dtype=torch.bfloat16) for i in range(12)];rows=[];reference=None
 # Compile-free eager GEMM warmup.
 for _ in range(10):torch.mm(x,cache[0,:2048*2048].view(2048,2048))
 torch.cuda.synchronize()
 for repeat in range(3):
  for ready in ([False,True] if repeat%2==0 else [True,False]):
   q=PriorityH2DScheduler(cache,profile=True)
   for i in range(12):q.enqueue_demand(i,i,host[i:i+1])
   q.synchronize()
   for i in range(4):q.enqueue_demand(i,100+i,host[i:i+1])
   outputs=[None]*12;spans=[];metrics=dict(waits=0,ready_before_first_wait=0)
   for i in expert_order([(i,) for i in range(12)],q,ready,metrics):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
    w=cache[i,:2048*2048].view(2048,2048)
    for _ in range(8):y=torch.mm(x,w)
    outputs[i]=y;q.record_slot_use(i);b.record();spans.append((a,b))
   q.close();torch.cuda.synchronize();out=torch.stack(outputs)
   if reference is None:reference=out.clone()
   else:assert torch.equal(reference,out)
   h=[(q.trace_origin.elapsed_time(t.begin),q.trace_origin.elapsed_time(t.done)) for t in q.trace if t.key>=100]
   c=[(q.trace_origin.elapsed_time(a),q.trace_origin.elapsed_time(b)) for a,b in spans]
   total=union_ms(h);hidden=intersection_ms(h,c)
   rows.append(dict(repeat=repeat,ready_first=ready,H2D_ms=total,hidden_ms=hidden,exposed_ms=total-hidden,metrics=metrics))
 med={str(k):statistics.median(r['exposed_ms'] for r in rows if r['ready_first']==k) for k in [False,True]}
 assert med['True']<=med['False']+.1,med
 result=dict(status='PASS',numerical_parity='bitwise',rows=rows,median_exposed_ms=med,tolerance_ms=.1,attribution='CUDA copy events intersect eager GEMM event windows; final kernel-level Nsight attribution remains M16')
 path=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004/M10_ready_exposure.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
 import sys
 unit()
 if '--gpu' in sys.argv:gpu()
