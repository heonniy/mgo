"""Diagnostic-only wrappers: CPU call spans and complete MoE spans, no new sync."""
import time
import torch
from . import decode_runtime as module


class GranularDiagnostic:
    def __init__(self,rt):
        self.rt=rt;self.current={};self.restores=[]
        def wrap(obj,name,label):
            old=getattr(obj,name);self.restores.append((obj,name,old))
            def measured(*args,**kwargs):
                begin=time.perf_counter_ns()
                with torch.profiler.record_function('pcie.'+label):result=old(*args,**kwargs)
                self.current[label]=[begin,time.perf_counter_ns()]
                return result
            setattr(obj,name,measured)
        wrap(rt.metadata,'collect','metadata_call')
        wrap(module,'gather_global_routes','metadata_call')
        wrap(rt.controller,'plan_current','controller_call')
        wrap(module,'plan_rank_partial_layout','layout_cpu_call')
        wrap(module,'pack_rank_partial_layout','index_pack_call')
        wrap(rt,'compute','executor_call')
        old=rt.execute;self.restores.append((rt,'execute',old))
        def execute(*args,**kwargs):
            self.current={};event=rt.index;begin=time.perf_counter_ns()
            with torch.profiler.record_function('pcie.moe_runtime'):result=old(*args,**kwargs)
            end=time.perf_counter_ns()
            trace=rt.pcie_phase_trace[-1];assert trace['event']==event
            trace.update(self.current);trace['moe_runtime']=[begin,end]
            trace['granular_notes']='CPU call spans can contain communication/device waits; nested intervals must not be summed. Complete MoE excludes attention/router. Controller call excludes state capture.'
            return result
        rt.execute=execute

    def close(self):
        for obj,name,old in reversed(self.restores):setattr(obj,name,old)
        self.restores=[]
