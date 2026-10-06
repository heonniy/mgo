"""Process-local diagnostic hooks. Unchanged arithmetic and ready-first traversal."""
import json,time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch
import torch.distributed as dist
from mgo_v2.ready_compute import expert_order
from mgo_v2.fused_transport import ForwardPacket

class Recorder:
 def __init__(self,rt):self.rt=rt;self.rows=[];self.serial=0
 @contextmanager
 def phase(self,name,**meta):
  if self.rt.index<48:yield;return
  serial=self.serial;self.serial+=1;event=self.rt.index;label=f'b2|{event}|{serial}|{name}'
  before=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW);torch.cuda.nvtx.range_push(label)
  try:yield
  finally:
   torch.cuda.nvtx.range_pop();after=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
   self.rows.append(dict(event=event,layer=event%48,step=event//48-1,serial=serial,phase=name,host_start_raw_ns=before,host_end_raw_ns=after,**meta))
 def write(self,path):path.write_text(json.dumps(dict(rank=self.rt.rank,clock='CLOCK_MONOTONIC_RAW',rows=self.rows),separators=(',',':'))+'\n')

def install(rt):
 import mgo_v2.decode_runtime as runtime_module
 rec=Recorder(rt);restore=[]
 def replace(obj,name,value):
  old=getattr(obj,name);setattr(obj,name,value);restore.append(lambda:setattr(obj,name,old));return old
 def wrap(obj,name,phase):
  old=getattr(obj,name)
  def call(*a,**kw):
   with rec.phase(phase):return old(*a,**kw)
  replace(obj,name,call)
 wrap(rt.controller,'plan_current','current_controller_cpu')
 wrap(rt,'apply_fetches','demand_h2d_enqueue_cpu')
 wrap(rt,'prefetch_next','prefetch_controller_cpu')
 old_wait=rt.h2d.wait_for_slot
 def wait(slot):
  ticket=rt.h2d.tickets[slot]
  with rec.phase('expert_ready_wait',slot=int(slot),key=int(ticket.key)):return old_wait(slot)
 replace(rt.h2d,'wait_for_slot',wait)
 old_compute=rt.compute
 def compute(self,packet,e,layer):
  if self.index<48:return old_compute(packet,e,layer)
  mode,received,_,rw=packet;assert mode=='current';groups=e['groups'];parts=[None]*len(groups)
  with rec.phase('expert_loop'):
   order=iter(expert_order(groups,self.h2d,self.args.ready_first,self.ready_metrics))
   while True:
    with rec.phase('expert_ready_select'):
     try:i=next(order)
     except StopIteration:break
    expert,rows,cols,slot=groups[i];assert self.keys[slot]==layer*128+expert
    meta=dict(expert=int(expert),rows=len(rows),slot=int(slot),key=int(layer*128+expert))
    executor=getattr(self,'graph_executor',None);consolidated=executor is not None and getattr(executor,'wrapper',False)
    with rec.phase('expert_gather',**meta):
     if consolidated:executor.gather(slot,received,rows)
     else:
      x=received[rows]
      if executor is None:
       w=self.cache[slot];gate=w[:1572864].view(768,2048);up=w[1572864:3145728].view(768,2048);down=w[3145728:].view(2048,768)
    with rec.phase('expert_compiled_kernel',**meta):
     part=executor.launch(slot,len(rows)) if consolidated else executor(slot,x) if executor is not None else self.kernel(x,gate,up,down)
    with rec.phase('expert_record_use',**meta):self.h2d.record_slot_use(slot)
    with rec.phase('expert_weight_partial',**meta):parts[i]=executor.weight(slot,part,rw,rows,cols) if consolidated else part*rw[rows,cols,None]
  return parts
 replace(rt,'compute',MethodType(compute,rt))
 old_forward=rt.transport.forward
 def forward(hidden,dense,e,async_op=False):
  if rt.index<48:return old_forward(hidden,dense,e,async_op)
  assert hidden.dtype==torch.bfloat16
  with rec.phase('forward_pack_gpu'):
   idx=e['send_idx'];ids=e['send_eids'];weights=dense[idx[:,None],ids.clamp_min(0)]*(ids>=0)
   n=len(idx);h=hidden.shape[-1];hb=h*2;payload=torch.empty((n,hb+24),device=hidden.device,dtype=torch.uint8)
   payload[:,:hb]=hidden[idx].contiguous().view(torch.uint8).view(n,hb)
   payload[:,hb:hb+16]=weights.contiguous().view(torch.uint8).view(n,16);payload[:,hb+16:]=ids.to(torch.uint8)
  with rec.phase('forward_collective_host_call'):raw,work=rt.transport.exchange(payload,e['send_counts'],e['recv_counts'],async_op)
  rt.transport.forward_bytes+=(sum(e['send_counts'])-e['send_counts'][rt.rank])*(hb+24)
  return ForwardPacket(raw,payload,work,h,hidden.dtype)
 replace(rt.transport,'forward',forward)
 old_finish=ForwardPacket.finish
 def finish(self):
  if rt.index<48:return old_finish(self)
  with rec.phase('forward_work_wait_cpu'):
   if self.work is not None:self.work.wait();self.work=None
  with rec.phase('forward_unpack_gpu'):
   self.send=None;n=self.raw.shape[0];hb=self.hidden_size*2
   hidden=self.raw[:,:hb].contiguous().view(self.dtype).view(n,self.hidden_size)
   weights=self.raw[:,hb:hb+16].contiguous().view(self.dtype).view(n,8);ids=self.raw[:,hb+16:hb+24].contiguous()
  return hidden,weights,ids
 replace(ForwardPacket,'finish',finish)
 old_combine=runtime_module.combine_rank_partials
 def combine(transport,hidden,parts,event,accumulation=torch.float32,unique_rows=False):
  if rt.index<48:return old_combine(transport,hidden,parts,event,accumulation,unique_rows)
  assert accumulation==torch.bfloat16 and unique_rows and event['unique_combine_validated']
  from mgo_v2.unique_combine import add_unique_rows_
  with rec.phase('return_partial_build_gpu'):
   partial=torch.zeros((sum(event['recv_counts']),hidden.shape[-1]),device=hidden.device,dtype=accumulation)
   for (_,rows,_,_),values in zip(event['groups'],parts):add_unique_rows_(partial,rows,values)
  with rec.phase('return_collective_host_call'):returned,_=transport.exchange(partial,event['recv_counts'],event['send_counts'])
  transport.return_bytes+=(sum(event['recv_counts'])-event['recv_counts'][transport.rank])*hidden.shape[-1]*partial.element_size()
  with rec.phase('return_output_combine_gpu'):
   output=torch.zeros_like(hidden,dtype=accumulation);offset=0
   for count in event['send_counts']:
    add_unique_rows_(output,event['send_idx'][offset:offset+count],returned[offset:offset+count]);offset+=count
   output=output.to(hidden.dtype)
  return output
 replace(runtime_module,'combine_rank_partials',combine)
 def uninstall():
  for f in reversed(restore):f()
 return rec,uninstall
