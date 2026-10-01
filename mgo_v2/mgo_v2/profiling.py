"""Optional CUDA collective measurements, excluded from default execution."""
from collections import defaultdict

import torch


class CollectiveStats:
    """Count submitted peer payload and time CUDA intervals around collectives.

    Bytes exclude local/self traffic, NCCL protocol headers and internal
    algorithms. Intervals include stream/launch waits; they are not sums of
    NCCL kernel durations. Use an Nsight trace for overlap and kernel timing.
    """

    def __init__(self):
        self.payload_bytes = defaultdict(int)
        self.events = []

    def call(self, kind, peer_bytes, operation):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.nvtx.range(kind):
            start.record()
            operation()
            end.record()
        self.payload_bytes[kind] += int(peer_bytes)
        self.events.append((kind, start, end))

    def summary(self):
        torch.cuda.synchronize()
        durations = defaultdict(float)
        calls = defaultdict(int)
        for kind, start, end in self.events:
            durations[kind] += start.elapsed_time(end)
            calls[kind] += 1
        return {"peer_payload_tx_bytes": dict(self.payload_bytes),
                "cuda_interval_ms": dict(durations), "calls": dict(calls)}
