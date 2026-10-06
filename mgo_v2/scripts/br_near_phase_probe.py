"""Opt-in stream-span diagnostics; no new waits or barriers in execution."""
import time
import torch
from mgo_v2.pinned_h2d import PriorityH2DScheduler

class PhaseProbe:
 def __init__(self,rt):
  self.rt=rt;self.spans=[];self.cpu=[];self.work=[];self.originals=[]
  rt.h2d.close()
  rt.h2d=PriorityH2DScheduler(rt.cache,profile=True,direct_pinned=True,cpu_team=rt.args.staging_cpu_team)
  self.origin=rt.h2d.trace_origin
  for obj,name,label in [(rt,'execute','moe'),(rt,'compute','compute'),(rt.metadata,'collect','metadata'),(rt.h2d,'wait_for_slot','h2d_wait')]:self.wrap_span(obj,name,label)
  for name in ['plan_current','plan_prefetch_next']:self.wrap_cpu(rt.controller,name)
  old=rt.plan_event
  def plan(*args,**kwargs):
   e=old(*args,**kwargs)
   if rt.index>=48:self.work.append(dict(event=rt.index,expert_rows=sum(len(g[1]) for g in e['groups']),experts=len(e['groups']),forward_remote_packets=sum(e['send_counts'])-e['send_counts'][rt.rank]))
   return e
  self.install(rt,'plan_event',plan)
  for name in ['enqueue_demand','enqueue_prefetch']:
   old_enqueue=getattr(rt.h2d,name)
   def enqueue(*args,_old=old_enqueue,_kind=name,**kwargs):
    t=_old(*args,**kwargs)
    if not hasattr(t,'probe_event'):t.probe_event=rt.index;t.probe_kind=_kind
    return t
   self.install(rt.h2d,name,enqueue)
  exchange=rt.transport.exchange
  probe=self
  def traced_exchange(*args,**kwargs):
   if rt.index<48:return exchange(*args,**kwargs)
   token=probe.begin('forward' if rt.transport.calls%2==0 else 'return')
   y,work=exchange(*args,**kwargs)
   if work is None:probe.end(token)
   else:work=WorkProxy(work,probe,token)
   return y,work
  self.install(rt.transport,'exchange',traced_exchange)
 def install(self,obj,name,new):
  self.originals.append((obj,name,getattr(obj,name)));setattr(obj,name,new)
 def begin(self,name):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  token=[self.rt.index,name,a,b,time.perf_counter(),None];self.spans.append(token);return token
 def end(self,token):
  token[3].record();token[5]=time.perf_counter()
 def wrap_span(self,obj,name,label):
  old=getattr(obj,name)
  def wrapped(*args,**kwargs):
   if self.rt.index<48:return old(*args,**kwargs)
   token=self.begin(label)
   try:return old(*args,**kwargs)
   finally:self.end(token)
  self.install(obj,name,wrapped)
 def wrap_cpu(self,obj,name):
  old=getattr(obj,name)
  def wrapped(*args,**kwargs):
   if self.rt.index<48:return old(*args,**kwargs)
   event=self.rt.index;t=time.perf_counter()
   try:return old(*args,**kwargs)
   finally:self.cpu.append(dict(event=event,name=name,ms=(time.perf_counter()-t)*1000))
  self.install(obj,name,wrapped)
 def finish(self):
  self.rt.h2d.synchronize();torch.cuda.synchronize()
  spans=[]
  for event,name,a,b,t0,t1 in self.spans:
   assert t1 is not None
   spans.append(dict(event=event,name=name,start_ms=self.origin.elapsed_time(a),end_ms=self.origin.elapsed_time(b),ms=a.elapsed_time(b),host_ms=(t1-t0)*1000))
  copies=[]
  for t in self.rt.h2d.trace:
   if t.probe_event>=48:copies.append(dict(event=t.probe_event,kind=t.probe_kind,key=t.key,slot=t.slot,start_ms=self.origin.elapsed_time(t.begin),end_ms=self.origin.elapsed_time(t.done),ms=t.begin.elapsed_time(t.done)))
  result=dict(spans=spans,copies=copies,cpu=self.cpu,work=self.work)
  for obj,name,old in reversed(self.originals):setattr(obj,name,old)
  self.originals.clear();return result

class WorkProxy:
 def __init__(self,work,probe,token):self.work=work;self.probe=probe;self.token=token
 def wait(self,*args,**kwargs):
  result=self.work.wait(*args,**kwargs);self.probe.end(self.token);return result
