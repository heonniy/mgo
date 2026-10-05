"""Diagnostic-only subdivision of the existing decode expert loop.

Arithmetic, traversal and return positions mirror DecodeOffloadRuntime.compute.
The worker validates the instrumented output against its unmodified warmup.
All labels are CPU call intervals, never GPU kernel duration estimates.
"""
from types import MethodType
import torch
from mgo_v2.ready_compute import expert_order


def install(rt,diag):
    restore=[]
    original_compute=rt.compute
    def compute(self,packet,event,layer):
        if self.index<48 or not (getattr(self.args,'streaming',False) or getattr(self.args,'fused',False)):
            return original_compute(packet,event,layer)
        mode,received,_,rw=packet
        assert mode=='current' and self.args.phase=='MEASURE'
        groups=event['groups'];parts=[None]*len(groups)
        order=iter(expert_order(groups,self.h2d,getattr(self.args,'ready_first',False),self.ready_metrics))
        while True:
            try:
                with diag.phase('expert.ready_select'):i=next(order)
            except StopIteration:break
            expert,rows,cols,slot=groups[i]
            assert self.keys[slot]==layer*128+expert
            with diag.phase('expert.input_and_weight_views'):
                w=self.cache[slot];x=received[rows]
                gate=w[:1572864].view(768,2048)
                up=w[1572864:3145728].view(768,2048)
                down=w[3145728:].view(2048,768)
            with diag.phase('expert.kernel_call'):part=self.kernel(x,gate,up,down)
            with diag.phase('expert.slot_use'):self.h2d.record_slot_use(slot)
            with diag.phase('expert.routing_weight_lookup'):weight=rw[rows,cols,None]
            with diag.phase('expert.weight_multiply'):parts[i]=part*weight
        if getattr(self.args,'fused',False) and self.index>=48:return parts
        return torch.cat(parts) if parts else received.new_empty((0,2048))
    rt.compute=MethodType(compute,rt);restore.append(lambda:setattr(rt,'compute',original_compute))
    for name in ('ready','wait_for_slot','_submitted'):
        original=getattr(rt.h2d,name)
        def call(*args,_original=original,_name=name,**kwargs):
            with diag.phase('scheduler.'+_name):return _original(*args,**kwargs)
        setattr(rt.h2d,name,call)
        restore.append(lambda name=name,original=original:setattr(rt.h2d,name,original))
    # Track actual host event API calls on both main/staging threads. No new
    # CUDA events or synchronization are introduced by these wrappers.
    for cls,names in ((torch.cuda.Event,('record','query','synchronize')),
                      (torch.cuda.Stream,('wait_event','synchronize'))):
        for name in names:
            original=getattr(cls,name);label='cuda.'+cls.__name__+'.'+name
            def call(*args,_original=original,_label=label,**kwargs):
                with diag.phase(_label):return _original(*args,**kwargs)
            setattr(cls,name,call)
            restore.append(lambda cls=cls,name=name,original=original:setattr(cls,name,original))
    def uninstall():
        for action in reversed(restore):action()
    return uninstall


def install_kernel_only(rt,diag):
    """One wrapper around the unchanged compiled callable; no copied loop."""
    original=rt.kernel
    def call(*args,**kwargs):
        with diag.phase('expert.kernel_call'):return original(*args,**kwargs)
    rt.kernel=call
    return lambda:setattr(rt,'kernel',original)
