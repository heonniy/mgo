"""CoSLoT-style bounded pinned staging for expert H2D.

The file-backed expert store stays pageable. A small pinned double-buffer is
used as the DMA source, mirroring CoSLoT/Archer's pinned staging semantics
without pinning the entire model. H2D runs on a dedicated CUDA stream.
Per-slot compute/fetch events prevent cache-slot overwrite races.
"""
from __future__ import annotations
import torch

def copy_expert_to_stage(stage: torch.Tensor, tensors) -> int:
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

class PinnedH2DCache:
    def __init__(self,cache:torch.Tensor,stage_count:int=2):
        if not cache.is_cuda:raise ValueError("cache must be CUDA")
        if stage_count<2:raise ValueError("stage_count must be >=2")
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
            self.cache[slot].copy_(stage,non_blocking=True)
            self.fetch_done[slot].record(self.h2d_stream);self.stage_done[sid].record(self.h2d_stream)
        self.stage_busy[sid]=True;self.fetch_valid[slot]=True
    def wait_for_slot(self,slot:int):
        slot=int(slot)
        if self.fetch_valid[slot]:torch.cuda.current_stream(device=self.cache.device).wait_event(self.fetch_done[slot])
    def record_slot_use(self,slot:int):
        slot=int(slot);self.compute_done[slot].record(torch.cuda.current_stream(device=self.cache.device));self.compute_valid[slot]=True
    def synchronize(self):self.h2d_stream.synchronize()
