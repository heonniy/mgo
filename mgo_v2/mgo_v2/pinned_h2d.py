"""CoSLoT-style bounded pinned staging for expert H2D.

The file-backed expert store stays pageable. A small pinned double-buffer is
used as the DMA source, mirroring CoSLoT/Archer's pinned staging semantics
without pinning the entire model. H2D runs on a dedicated CUDA stream.
Per-slot compute/fetch events prevent cache-slot overwrite races.
"""
from __future__ import annotations
import torch
import ctypes

_MEMMOVE=ctypes.CDLL(None).memmove
_MEMMOVE.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
_MEMMOVE.restype=ctypes.c_void_p

def copy_expert_to_stage(stage: torch.Tensor, tensors, backend="torch") -> int:
    if backend not in ("torch","memmove"):
        raise ValueError("unknown staging backend")
    if backend=="memmove":
        # Validate every source and destination before raw pointer access.
        # Tensor references stay alive throughout each GIL-releasing C call.
        tensors=tuple(tensors)
        if stage.device.type!="cpu" or stage.ndim!=1 or not stage.is_contiguous():
            raise ValueError("native staging requires a contiguous one-dimensional CPU destination")
        if stage.dtype!=torch.bfloat16 or any(t.device.type!="cpu" or t.dtype!=stage.dtype or not t.is_contiguous() for t in tensors):
            raise ValueError("native staging requires contiguous CPU BF16 sources")
        sizes=[t.numel() for t in tensors]
        if sum(sizes)!=stage.numel():
            raise ValueError("expert size does not match staging capacity")
        pos=0;itemsize=stage.element_size();base=stage.data_ptr()
        for tensor,size in zip(tensors,sizes):
            if size:_MEMMOVE(base+pos*itemsize,tensor.data_ptr(),size*itemsize)
            pos+=size
        return pos
    if stage.device.type != "cpu":
        raise ValueError("stage must be CPU")
    pos=0
    for tensor in tensors:
        if tensor.device.type!="cpu":
            raise ValueError("expert backing must be CPU")
        flat=tensor.reshape(-1);end=pos+flat.numel()
        if end>stage.numel():
            raise RuntimeError("expert exceeds pinned staging capacity")
        stage[pos:end].copy_(flat);pos=end
    if pos!=stage.numel():
        raise RuntimeError(f"expert size mismatch copied={pos} expected={stage.numel()}")
    return pos

def build_full_pinned_expert_store(experts, bytes_per_expert: int):
    """Materialize one full rank-private pinned expert copy.

    This is deliberately simple for the R4 experiment: every rank owns its own
    page-locked copy of all experts.  Returned expert sources are one flat,
    contiguous pinned BF16 tensor per expert, so the H2D scheduler can DMA
    directly without the pageable/file-backed -> staging memcpy.
    """
    if bytes_per_expert % 2:
        raise ValueError("BF16 expert size must be even")
    rows=len(experts);elems=bytes_per_expert//2
    try:
        pool=torch.empty((rows,elems),dtype=torch.bfloat16,device="cpu",pin_memory=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"failed to allocate full pinned expert store: {rows*bytes_per_expert/2**30:.2f} GiB"
        ) from exc
    if not pool.is_pinned():
        raise RuntimeError("full expert store is not pinned")
    for key,tensors in enumerate(experts):
        copy_expert_to_stage(pool[key],tensors,"torch")
    sources=[(pool[key],) for key in range(rows)]
    return pool,sources

class PinnedH2DCache:
    def __init__(self,cache:torch.Tensor,stage_count:int=2):
        if not cache.is_cuda:raise ValueError("cache must be CUDA")
        if stage_count<2:raise ValueError("stage_count must be >=2")
        self.trace=None
        self.cache=cache;self.stage_count=int(stage_count)
        self.h2d_stream=torch.cuda.Stream(device=cache.device)
        self.stages=[torch.empty(cache.shape[1],dtype=cache.dtype,device="cpu",pin_memory=True) for _ in range(self.stage_count)]
        if not all(x.is_pinned() for x in self.stages):raise RuntimeError("pinned allocation failed")
        self.stage_done=[torch.cuda.Event(enable_timing=False) for _ in self.stages];self.stage_busy=[False]*self.stage_count
        self.fetch_done=[torch.cuda.Event(enable_timing=False) for _ in range(cache.shape[0])];self.fetch_valid=[False]*cache.shape[0]
        self.compute_done=[torch.cuda.Event(enable_timing=False) for _ in range(cache.shape[0])];self.compute_valid=[False]*cache.shape[0]
        self.cursor=0;self.bytes_per_expert=cache.shape[1]*cache.element_size()
    @property
    def pinned_bytes(self):return self.stage_count*self.bytes_per_expert
    def enqueue(self,slot:int,tensors):
        slot=int(slot);sid=self.cursor;self.cursor=(self.cursor+1)%self.stage_count
        if self.stage_busy[sid]:self.stage_done[sid].synchronize()
        stage=self.stages[sid];copy_expert_to_stage(stage,tensors)
        with torch.cuda.stream(self.h2d_stream):
            if self.compute_valid[slot]:self.h2d_stream.wait_event(self.compute_done[slot])
            token=self.trace.copy_start(self.h2d_stream,slot) if self.trace else None
            self.cache[slot].copy_(stage,non_blocking=True)
            if self.trace:self.trace.copy_end(token,self.h2d_stream)
            self.fetch_done[slot].record(self.h2d_stream);self.stage_done[sid].record(self.h2d_stream)
        self.stage_busy[sid]=True;self.fetch_valid[slot]=True
    def wait_for_slot(self,slot:int):
        slot=int(slot)
        if self.fetch_valid[slot]:torch.cuda.current_stream(device=self.cache.device).wait_event(self.fetch_done[slot])
    def record_slot_use(self,slot:int):
        slot=int(slot);self.compute_done[slot].record(torch.cuda.current_stream(device=self.cache.device));self.compute_valid[slot]=True
    def synchronize(self):self.h2d_stream.synchronize()

# Background staging is an explicit alternative; the synchronous path above
# remains the frozen baseline and diagnostic implementation.
import threading
import time
from collections import deque

class CopyTicket:
    def __init__(self, slot, key, tensors, urgent, profile, event_index=None):
        from .prefetch import TransferState
        self.slot=int(slot);self.key=int(key);self.tensors=tensors;self.valid=True
        self.event_index=event_index
        self.state=TransferState(key)
        if urgent:self.state.demand()
        self.done=torch.cuda.Event(enable_timing=profile)
        self.stream_enter=torch.cuda.Event(enable_timing=True) if profile else None
        self.begin=torch.cuda.Event(enable_timing=True) if profile else None
        self.submitted=False;self.staging=False;self.counted=True
        self.previous_copy=None;self.previous_compute=None
        self.queued_at=time.perf_counter();self.submitted_at=None;self.staging_started_at=None;self.staging_finished_at=None

class PriorityH2DScheduler:
    """One CPU staging worker, bounded pinned stages, urgent-before-background.

    Stage reuse waits for DMA. Slot overwrite waits for both prior fetch and
    prior compute. A promoted reservation reuses its ticket/copy rather than
    issuing a second H2D. Physical readiness is not replicated controller state.
    """
    def __init__(self,cache,stage_count=2,profile=False,autostart=True,staging_backend="torch",cpu_team=None,direct_pinned=False):
        if staging_backend not in ("torch","memmove"):raise ValueError("unknown staging backend")
        self.staging_backend=staging_backend;self.direct_pinned=bool(direct_pinned)
        self.cpu_team=cpu_team;self.cpu_team_receipt=None;self.startup_done=threading.Event()
        if cpu_team is not None and staging_backend!="torch":raise ValueError("fixed copy team requires torch staging")
        if not cache.is_cuda or stage_count<2:raise ValueError('CUDA arena and >=2 stages required')
        self.cache=cache;self.profile=profile;self.device=cache.device
        self.h2d_stream=torch.cuda.Stream(device=self.device)
        self.stages=[torch.empty(cache.shape[1],dtype=cache.dtype,device='cpu',pin_memory=True) for _ in range(stage_count)]
        self.stage_done=[None]*stage_count;self.cursor=0
        self.tickets={};self.compute_done={};self.urgent=deque();self.background=deque()
        self.cv=threading.Condition();self.pending=0;self.error=None;self.stopping=False;self.thread=None
        self.bytes_per_expert=cache.shape[1]*cache.element_size()
        self.metrics=dict(copies=0,bytes=0,background_copies=0,urgent_copies=0,canceled=0,escalated=0,restaged=0)
        self.trace=[];self.trace_origin=torch.cuda.Event(enable_timing=True) if profile else None
        if profile:self.trace_origin.record();self.trace_origin.synchronize()
        if autostart:self.start()
    @property
    def pinned_bytes(self):return len(self.stages)*self.bytes_per_expert
    def start(self):
        if self.thread is not None:return
        self.thread=threading.Thread(target=self._worker,name='expert-staging',daemon=True);self.thread.start()
        if self.cpu_team is not None:
            if not self.startup_done.wait(30):raise TimeoutError('staging CPU team initialization timeout')
            self._check()
    def _check(self):
        if self.error is not None:raise RuntimeError('expert staging worker failed') from self.error
    def _retire_pending(self,t):
        if t.counted:self.pending-=1;t.counted=False
    def _enqueue(self,slot,key,tensors,urgent,event_index=None):
        with self.cv:
            self._check()
            if self.stopping:raise RuntimeError('scheduler closed')
            old=self.tickets.get(slot)
            if old is not None and old.key==key and old.valid and old.state.state!='EMPTY':
                if urgent and not old.state.urgent:
                    old.state.demand();self.metrics['escalated']+=1
                    if not old.submitted:self.urgent.append(old)
                self.cv.notify_all();return old
            t=CopyTicket(slot,key,tensors,urgent,self.profile,event_index)
            if old is not None:
                if old.submitted or old.state.state=='INFLIGHT':t.previous_copy=old.done
                elif old.state.state=='QUEUED':
                    if old.state.urgent:raise RuntimeError('overwriting unsubmitted current demand')
                    # Superseded reservation: don't enqueue its staged bytes.
                    old.state.state='EMPTY';self._retire_pending(old);self.metrics['canceled']+=1
            t.previous_compute=self.compute_done.get(slot)
            self.tickets[slot]=t;self.pending+=1
            (self.urgent if urgent else self.background).append(t);self.cv.notify_all();return t
    def enqueue_demand(self,slot,key,tensors,event_index=None):return self._enqueue(slot,key,tensors,True,event_index)
    def enqueue_prefetch(self,slot,key,tensors):return self._enqueue(slot,key,tensors,False)
    def promote(self,slot,key):
        with self.cv:
            t=self.tickets[slot]
            if t.key!=key or t.state.state=='EMPTY':raise RuntimeError('missing promoted transfer')
            if not t.state.urgent:
                t.state.demand();self.metrics['escalated']+=1
                if not t.submitted:self.urgent.append(t)
            self.cv.notify_all();return t
    def invalidate(self,slot,key):
        # Retiring a logical role must invalidate deduplication even though its
        # physical bytes remain until overwrite. Keep hazard events intact.
        with self.cv:
            t=self.tickets.get(slot)
            if t is not None and t.key==key:t.valid=False
    def cancel_if_queued(self,slot,key):
        with self.cv:
            t=self.tickets.get(slot)
            if t is None or t.key!=key or t.state.urgent or t.state.state!='QUEUED':return False
            t.valid=False;t.state.discard();self._retire_pending(t);self.metrics['canceled']+=1;self.cv.notify_all();return True
    def discard(self,slot,key):
        with self.cv:
            t=self.tickets.get(slot)
            if t is None or t.key!=key:return
            queued=t.state.state=='QUEUED'
            t.valid=False;t.state.discard()
            if queued:self._retire_pending(t);self.metrics['canceled']+=1
            self.cv.notify_all()
    def _pop(self):
        for queue in (self.urgent,self.background):
            while queue:
                t=queue.popleft()
                if t.state.state=='QUEUED' and not t.staging:
                    t.staging=True;return t
        return None
    def _worker(self):
        try:
            torch.cuda.set_device(self.device)
            if self.cpu_team is not None and not self.direct_pinned:
                from .staging_cpu_team import initialize
                self.cpu_team_receipt=initialize(self.stages[0],self.cpu_team)
            self.startup_done.set()
            while True:
                with self.cv:
                    while not self.pending and not self.stopping:self.cv.wait()
                    if self.stopping and not self.pending:return
                sid=self.cursor
                if not self.direct_pinned and self.stage_done[sid] is not None:self.stage_done[sid].synchronize()
                with self.cv:
                    t=self._pop()
                    if t is None:continue
                direct_source=None
                if self.direct_pinned:
                    tensors=tuple(t.tensors)
                    if len(tensors)!=1:
                        raise RuntimeError("direct-pinned source must be one flat expert tensor")
                    direct_source=tensors[0]
                    if (direct_source.device.type!="cpu" or not direct_source.is_pinned()
                        or not direct_source.is_contiguous() or direct_source.dtype!=self.cache.dtype
                        or direct_source.numel()!=self.cache.shape[1]):
                        raise RuntimeError("invalid direct-pinned expert source")
                    if self.profile:
                        now=time.perf_counter();t.staging_started_at=now;t.staging_finished_at=now
                else:
                    if self.profile:t.staging_started_at=time.perf_counter()
                    copy_expert_to_stage(self.stages[sid],t.tensors,self.staging_backend)
                    if self.profile:t.staging_finished_at=time.perf_counter()
                with self.cv:
                    t.staging=False
                    if t.state.state=='EMPTY':self.cv.notify_all();continue
                    if (not self.direct_pinned and not t.state.urgent
                        and any(x.state.state=='QUEUED' for x in self.urgent)):
                        # Urgent arrival during CPU staging: requeue speculative
                        # bytes instead of placing them ahead on the DMA stream.
                        self.background.appendleft(t);self.metrics['restaged']+=1;continue
                    t.state.start()
                # Direct-pinned mode retains the same queue, priority, slot
                # hazards and completion events, but removes CPU staging.
                with torch.cuda.stream(self.h2d_stream):
                    if t.stream_enter is not None:t.stream_enter.record(self.h2d_stream)
                    if t.previous_copy is not None:self.h2d_stream.wait_event(t.previous_copy)
                    if t.previous_compute is not None:self.h2d_stream.wait_event(t.previous_compute)
                    if t.begin is not None:t.begin.record(self.h2d_stream)
                    self.cache[t.slot].copy_(direct_source if self.direct_pinned else self.stages[sid],non_blocking=True)
                    t.done.record(self.h2d_stream)
                with self.cv:
                    t.submitted=True;t.submitted_at=time.perf_counter();self._retire_pending(t)
                    if not self.direct_pinned:
                        self.stage_done[sid]=t.done;self.cursor=(sid+1)%len(self.stages)
                    self.metrics['copies']+=1;self.metrics['bytes']+=self.bytes_per_expert
                    self.metrics['urgent_copies' if t.state.urgent else 'background_copies']+=1
                    if self.profile:self.trace.append(t)
                    t.tensors=None;self.cv.notify_all()
        except BaseException as exc:
            with self.cv:self.error=exc;self.cv.notify_all()
            self.startup_done.set()
    def _submitted(self,t,timeout=120):
        deadline=time.monotonic()+timeout
        with self.cv:
            while not t.submitted:
                self._check()
                if t.state.state=='EMPTY':raise RuntimeError('waiting on canceled copy')
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('H2D submission timeout')
                self.cv.wait(min(remaining,1))
    def ready(self,slot):
        with self.cv:
            self._check();t=self.tickets.get(slot)
            if t is None or not t.valid:return False
            if t.state.state=='READY':return True
            if not t.submitted:return False
        complete=t.done.query()
        with self.cv:
            self._check()
            if self.tickets.get(slot) is not t or not t.valid:return False
            if complete and t.state.state=='INFLIGHT':t.state.complete()
            return t.state.state=='READY'
    def ready_many(self,slots):
        # One critical section for a consistent ticket snapshot and completion
        # updates. query() is nonblocking; no wait occurs under this lock.
        with self.cv:
            self._check();mask=[]
            for slot in slots:
                t=self.tickets.get(slot)
                if t is None or not t.valid:mask.append(False);continue
                if t.state.state=='INFLIGHT' and t.submitted and t.done.query():t.state.complete()
                mask.append(t.state.state=='READY')
            return mask
    def record_slots_use(self,slots):
        event=torch.cuda.Event();event.record(torch.cuda.current_stream(device=self.device))
        with self.cv:
            for slot in slots:self.compute_done[slot]=event
    def wait_for_slot(self,slot):
        t=self.tickets[slot];self._submitted(t)
        torch.cuda.current_stream(device=self.device).wait_event(t.done)
    def wait_slots(self,slots,host=False):
        for slot in slots:
            t=self.tickets[slot];self._submitted(t)
            if host:
                t.done.synchronize()
                with self.cv:
                    if t.state.state=='INFLIGHT':t.state.complete()
            else:torch.cuda.current_stream(device=self.device).wait_event(t.done)
    def record_slot_use(self,slot):
        event=torch.cuda.Event();event.record(torch.cuda.current_stream(device=self.device))
        with self.cv:self.compute_done[slot]=event
    def synchronize(self):
        deadline=time.monotonic()+120
        with self.cv:
            while self.pending:
                self._check()
                if time.monotonic()>deadline:raise TimeoutError('H2D queue drain timeout')
                self.cv.wait(.2)
            self._check()
        self.h2d_stream.synchronize()
    def close(self):
        try:self.synchronize()
        finally:
            with self.cv:self.stopping=True;self.cv.notify_all()
            if self.thread is not None:self.thread.join(timeout=5)
