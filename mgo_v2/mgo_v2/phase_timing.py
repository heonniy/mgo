"""Opt-in CUDA stream-boundary instrumentation; actual H2D DMA recorded separately."""
from contextlib import contextmanager
import time
import torch

def union_ms(intervals):
 total=0.;end=None
 for a,b in sorted(intervals):
  if b<a:raise ValueError('negative interval')
  total+=max(0.,b-max(a,end if end is not None else a));end=max(b,end if end is not None else b)
 return total

def intersection_ms(left,right):
 cuts=[]
 for a,b in left:
  for c,d in right:
   if min(b,d)>max(a,c):cuts.append((max(a,c),min(b,d)))
 return union_ms(cuts)

class PhaseRecorder:
 def __init__(self):
  self.origin=torch.cuda.Event(enable_timing=True);self.origin.record();self.origin.synchronize()
  self.event=-1;self.records=[];self.copies=[]
 @contextmanager
 def span(self,name):
  if self.event<48:
   yield;return
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();start=time.perf_counter()
  torch.cuda.nvtx.range_push(name)
  try:yield
  finally:
   torch.cuda.nvtx.range_pop();b.record();self.records.append((self.event,name,start,time.perf_counter(),a,b))
 def copy_start(self,stream,slot):
  if self.event<48:return None
  a=torch.cuda.Event(enable_timing=True);a.record(stream);return self.event,slot,a
 def copy_end(self,token,stream):
  if token is None:return
  b=torch.cuda.Event(enable_timing=True);b.record(stream);self.copies.append((*token,b))
 def receipt(self):
  torch.cuda.synchronize();rows=[]
  for event in sorted({x[0] for x in self.records}):
   records=[x for x in self.records if x[0]==event]
   spans={n:(self.origin.elapsed_time(a),self.origin.elapsed_time(b)) for _,n,_,_,a,b in records}
   cpu={n:(end-start)*1000 for _,n,start,end,_,_ in records}
   whole=[spans['moe.layer']];parts=[v for k,v in spans.items() if k!='moe.layer']
   dma=[(self.origin.elapsed_time(a),self.origin.elapsed_time(b)) for e,s,a,b in self.copies if e==event]
   useful=[v for k,v in spans.items() if k in ('moe.dispatch','moe.expert_compute','moe.combine')]
   covered=intersection_ms(parts,whole);duration=union_ms(whole);unattributed=max(0.,duration-covered)
   assert abs(duration-covered-unattributed)<=.05
   rows.append(dict(event=event,span_ms=duration,phase_boundary_union_ms=covered,unattributed_ms=unattributed,reconciliation_error_ms=abs(duration-covered-unattributed),CPU_wall_ms=cpu,GPU_stream_boundaries_ms={k:b-a for k,(a,b) in spans.items()},H2D_DMA_ms=union_ms(dma),H2D_hidden_by_useful_boundaries_ms=intersection_ms(dma,useful)))
  return dict(status='PASS',events=rows,note='CUDA stream-boundary spans include submission gaps; they are not kernel-only attribution. H2D_DMA comes from copy-stream events around copies. Unattributed gaps are explicit. Nsight/CUPTI kernel-interval attribution is required at M16.',tolerance_ms=.05)
