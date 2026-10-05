"""Diagnostic-only wall/own-thread CPU phase accounting; no CUDA timing events.

Nested ranges are exclusive within a thread. Different threads overlap and
must never be summed into generation wall time. Wall minus thread CPU also
includes blocking/GIL/descheduling; it does not identify a particular wait.
"""
from contextlib import contextmanager
from collections import defaultdict
import gc
import threading
import time

class HostDiagnostics:
    def __init__(self, event_index):
        self.event_index=event_index
        self.local=threading.local()
        self.rows={}
        self.gc_rows=[]
        self.gc_start={}
        self.enabled=False

    @contextmanager
    def phase(self,name):
        if not self.enabled:
            yield
            return
        tid=threading.get_native_id()
        if not hasattr(self.local,'stack'):self.local.stack=[]
        stack=self.local.stack
        event=self.event_index()
        frame=[time.perf_counter_ns(),time.thread_time_ns(),0,0]
        stack.append(frame)
        try:yield
        finally:
            cpu=time.thread_time_ns()-frame[1];wall=time.perf_counter_ns()-frame[0]
            assert stack.pop() is frame
            if stack:stack[-1][2]+=wall;stack[-1][3]+=cpu
            key=(tid,threading.current_thread().name,event//48-1,name)
            row=self.rows.setdefault(key,[0,0,0,0,0])
            row[0]+=1;row[1]+=wall;row[2]+=cpu
            row[3]+=wall-frame[2];row[4]+=cpu-frame[3]

    def gc_callback(self,phase,info):
        if not self.enabled:return
        tid=threading.get_native_id()
        if phase=='start':self.gc_start[tid]=(time.perf_counter_ns(),time.thread_time_ns())
        elif tid in self.gc_start:
            wall,cpu=self.gc_start.pop(tid)
            self.gc_rows.append(dict(tid=tid,event_observed=self.event_index(),generation=info['generation'],wall_ns=time.perf_counter_ns()-wall,cpu_ns=time.thread_time_ns()-cpu,collected=info.get('collected'),uncollectable=info.get('uncollectable')))

    def start(self):
        assert not self.enabled
        self.enabled=True;gc.callbacks.append(self.gc_callback)

    def finish(self):
        gc.callbacks.remove(self.gc_callback);self.enabled=False
        return dict(phases=[dict(tid=k[0],thread=k[1],observed_step=k[2],phase=k[3],calls=v[0],inclusive_wall_ns=v[1],inclusive_cpu_ns=v[2],exclusive_wall_ns=v[3],exclusive_cpu_ns=v[4]) for k,v in sorted(self.rows.items())],gc=self.gc_rows,scope='Instrumented diagnostic only. Per-thread nested-exclusive intervals; do not sum threads. observed_step is main event observed on entry, not copy provenance. CPU time includes spin waits; non-CPU wall includes GIL, blocking and scheduling without distinguishing them.')
