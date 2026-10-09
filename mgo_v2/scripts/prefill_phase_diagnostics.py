"""Opt-in nested CPU/current-stream partition; no extra phase synchronization."""
from contextlib import contextmanager
from collections import defaultdict
import time
import numpy as np
import torch
import torch.distributed as dist

class PrefillDiagnostics:
 def __init__(self,rt):
  self.rt=rt;self.active=False;self.stack=[];self.rows=[];self.undo=[];self.hooks=[];self.bytes=defaultdict(int)
 def mark(self):
  event=torch.cuda.Event(enable_timing=True);event.record()
  now=time.perf_counter_ns();cpu=time.thread_time_ns()
  if self.rows:
   self.rows[-1]['wall_ns']=now-self.wall;self.rows[-1]['cpu_ns']=cpu-self.cpu
  self.rows.append(dict(event=event,layer=min(self.rt.index,47),phase=self.stack[-1] if self.stack else 'attention_dense_residual'))
  self.wall=now;self.cpu=cpu
 def begin(self,name):
  self.stack.append(name)
  if self.active:self.mark()
 def end(self):
  self.stack.pop()
  if self.active:self.mark()
 @contextmanager
 def phase(self,name):
  self.begin(name)
  try:yield
  finally:self.end()
 def start(self):
  self.rows=[];self.stack=[];self.active=True;self.mark()
 def stop(self):
  self.mark();self.active=False
 def wrap(self,obj,name,label):
  orig=getattr(obj,name)
  def call(*a,**kw):
   with self.phase(label):return orig(*a,**kw)
  setattr(obj,name,call);self.undo.append(lambda:setattr(obj,name,orig))
 def install(self,model):
  import mgo_v2.decode_runtime as dr
  import mgo_v2.runtime as runtime
  # Native NVTX scopes already surround dispatch, wait, compute and combine.
  orig=dr.nvtx_phase
  @contextmanager
  def nvtx(name):
   with self.phase(name):
    with orig(name):yield
  dr.nvtx_phase=nvtx;self.undo.append(lambda:setattr(dr,'nvtx_phase',orig))
  original=dr.gather_global_routes
  def gather(layer,selected,weights,probs,**kw):
   with self.phase('metadata_counts'):
    count=torch.tensor([len(selected)],device=selected.device,dtype=torch.int64);buf=[torch.empty_like(count) for _ in range(dist.get_world_size())];dist.all_gather(buf,count)
   with self.phase('device_to_host_counts'):counts=torch.cat(buf).tolist()
   def collect(x,cs,name):
    self.bytes[name]+=max(cs)*int(np.prod(x.shape[1:]))*x.element_size()
    with self.phase(name):return runtime._all_gather_padded(x,cs)
   sel=collect(selected.to(torch.int64),counts,'metadata_selected_ids')
   wei=collect(weights,counts,'metadata_routing_weights')
   tail=kw['probability_tail'];assert tail==128
   offsets=np.cumsum([0]+counts);start=max(0,sum(counts)-tail)
   cs=[max(0,int(offsets[r+1])-max(int(offsets[r]),start)) for r in range(len(counts))];rank=dist.get_rank();n=cs[rank]
   pro=collect((probs[-n:] if n else probs[:0]).float(),cs,'metadata_probability_history')
   with self.phase('device_to_host_controller_materialization'):
    ids=torch.cat(sel).cpu().numpy();ws=torch.cat(wei).float().cpu().numpy();ps=torch.cat(pro).float().cpu().numpy()
    origins=np.concatenate([np.full(n,r,dtype=np.int64) for r,n in enumerate(counts)])
   return runtime.GatheredRoutes(runtime.LayerRoutes(layer=layer,origin_ranks=origins,selected_experts=ids,routing_weights=ws,full_router_probs=ps),counts,sum(counts[:rank]))
  dr.gather_global_routes=gather;self.undo.append(lambda:setattr(dr,'gather_global_routes',original))
  self.wrap(self.rt.controller,'plan_current','placement_controller_cpu')
  self.wrap(dr,'plan_layout','layout_cpu')
  self.wrap(dr,'plan_rank_partial_layout','layout_cpu')
  self.wrap(dr,'pack_rank_partial_layout','layout_device_materialization')
  self.wrap(dr,'device_layout','layout_device_materialization')
  self.wrap(self.rt.h2d,'wait_slots','required_h2d_exposed_wait')
  self.wrap(dist,'barrier','global_barrier_wait')
  original_exchange=self.rt.transport.exchange
  def exchange(*a,**kw):
   label='return_token_a2a' if 'moe.return_a2a' in self.stack else 'forward_token_a2a_submit'
   with self.phase(label):return original_exchange(*a,**kw)
  self.rt.transport.exchange=exchange;self.undo.append(lambda:setattr(self.rt.transport,'exchange',original_exchange))
  # Gate projection, softmax/topk/normalization and reshaping precede execute.
  for block in model.model.layers:
   self.hooks.append(block.mlp.register_forward_pre_hook(lambda *a:self.begin('router_gate_compute')))
  original_execute=self.rt.execute
  def execute(*a,**kw):
   assert self.stack[-1]=='router_gate_compute';self.end()
   with self.phase('moe_other'):return original_execute(*a,**kw)
  self.rt.execute=execute;self.undo.append(lambda:setattr(self.rt,'execute',original_execute))
  self.rt.h2d.profile=True;self.rt.h2d.trace=[]
  self.rt.h2d.trace_origin=torch.cuda.Event(enable_timing=True);self.rt.h2d.trace_origin.record();self.rt.h2d.trace_origin.synchronize()
 def finish(self,wall_seconds):
  torch.cuda.synchronize();rows=[];totals=defaultdict(float)
  for a,b in zip(self.rows,self.rows[1:]):
   value=a['event'].elapsed_time(b['event'])/1000
   row={k:v for k,v in a.items() if k!='event'};row['stream_seconds']=value;rows.append(row);totals[a['phase']]+=value
  stream_total=sum(totals.values());residual=wall_seconds-stream_total
  assert abs(residual)<.05,(wall_seconds,stream_total,residual)
  totals['clock_boundary_residual']=residual
  copies=[]
  for t in self.rt.h2d.trace:
   copies.append(dict(layer=t.key//128,key=t.key,slot=t.slot,event_index=getattr(t,'event_index',None),bytes=self.rt.h2d.bytes_per_expert,service_seconds=t.begin.elapsed_time(t.done)/1000))
  assert sum(x['bytes'] for x in copies)==self.rt.h2d.metrics['bytes']
  self.rt.h2d.profile=False
  for hook in self.hooks:hook.remove()
  for undo in reversed(self.undo):undo()
  return dict(status='PASS',rank=dist.get_rank(),wall_seconds=wall_seconds,stream_seconds=stream_total,exclusive_stream_partition=totals,segments=rows,h2d_copies=copies,h2d_service_seconds=sum(x['service_seconds'] for x in copies),metadata_padded_send_bytes=dict(self.bytes),scope='Diagnostic current-stream completion spans include host submission gaps and peer waiting, not pure kernel times; H2D copy service is non-additive.')
