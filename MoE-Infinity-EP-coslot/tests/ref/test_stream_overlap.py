"""Phase 4 (stream parallelism) — fetch ‖ exec overlap (H9/H10/H11).

The C++ ExpertDispatcher fetches weights on ``h2d_streams_[gpu]`` and runs the
expert GEMM on ``exec_streams_[gpu]`` — two distinct non-blocking streams whose
only cross-stream synchronization is a per-slot CUDA event.  Hits push their
ExecTask with ``fetch_event=nullptr`` so a hit's compute NEVER waits on a miss
fetch.  This file verifies that contract three ways:

  1. STRUCT (always runs): grep the C++ to pin H9/H10 — distinct streams, hits
     push a null fetch_event, the exec worker waits only `if fetch_event!=null`,
     and there is no device-wide sync on the hot path.  A future change that
     serializes fetch and exec fails here.

  2. HARDWARE (runs on CUDA): reproduce the dispatcher's pattern with torch —
     an async H2D copy on stream A ‖ a GEMM on stream B — and measure that the
     concurrent wall-clock is well below the serial sum, i.e. they overlap on
     THIS gpu.  This is the mechanism the dispatcher relies on.

  3. PRODUCTION METRIC (gated): ``get_phase_times()[0]`` is ``fetch_wait_us`` —
     the CPU time the exec worker blocks on fetch_event.  Under real overlap it
     is ≪ the H2D transfer time.  ``overlap_ratio_from_phase_times`` computes the
     check; the end-to-end run needs a loaded model (register_expert), so the
     gated test documents how to read it from a benchmark.

Run:  pytest tests/ref/test_stream_overlap.py -q
"""
from __future__ import annotations

import os
import re

import pytest

ARCHER = "/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP-archer-coslot"
DISP_CPP = os.path.join(ARCHER, "core/parallel/expert_dispatcher.cpp")
DISP_H = os.path.join(ARCHER, "core/parallel/expert_dispatcher.h")


# ==================================================================
# 1. STRUCT — pin the stream-separation / hit-nonblocking invariants
# ==================================================================
def test_struct_streams_distinct_and_hits_nonblocking():
    cpp = open(DISP_CPP).read()
    h = open(DISP_H).read()

    # H9: two distinct stream pools exist.
    assert "h2d_streams_" in (cpp + h), "fetch stream pool missing"
    assert "exec_streams_" in (cpp + h), "exec stream pool missing"

    # H10: hits push ExecTask with a NULL fetch_event (no wait).
    assert re.search(r"PushExecTask\([^;]*nullptr\)", cpp), \
        "hit path must PushExecTask with fetch_event=nullptr"

    # H10: exec worker waits on the fetch event ONLY when it is non-null.
    assert re.search(r"if\s*\(\s*task\.fetch_event\s*!=\s*nullptr\s*\)", cpp), \
        "exec worker must guard cudaEventSynchronize on fetch_event != nullptr"

    # H9: cross-stream sync is per-slot event, not a device-wide barrier.
    assert "cudaStreamWaitEvent(h2d_streams_" in cpp, \
        "fetch overwrite must gate on the per-slot compute event"
    assert "cudaDeviceSynchronize" not in cpp, \
        "no device-wide sync allowed on the dispatcher hot path"


# ==================================================================
# 2. HARDWARE — H2D copy ‖ GEMM overlap on this GPU
# ==================================================================
def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_hardware_h2d_gemm_overlap():
    import time

    import torch

    dev = torch.device("cuda:0")
    h2d_stream = torch.cuda.Stream(dev)      # mimics dispatcher h2d_streams_
    exec_stream = torch.cuda.Stream(dev)     # mimics dispatcher exec_streams_

    # H2D workload: large pinned host buffer copied to device, repeated, so the
    # transfer takes several ms (DMA / copy engine).
    nbytes = 256 * 1024 * 1024               # 256 MiB
    host = torch.empty(nbytes // 4, dtype=torch.float32, pin_memory=True)
    dst = torch.empty_like(host, device=dev)
    n_copy = 24

    # GEMM workload (mimics the expert MoEMLP forward on exec_stream / SMs).
    n = 4096
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    n_gemm = 120

    def run_h2d():
        with torch.cuda.stream(h2d_stream):
            for _ in range(n_copy):
                dst.copy_(host, non_blocking=True)

    def run_gemm():
        with torch.cuda.stream(exec_stream):
            c = a
            for _ in range(n_gemm):
                c = torch.mm(c, b)
            return c

    def timed(fn):
        # Wall-clock between two full-device syncs captures async work on ANY
        # stream (recording events on the default stream would miss it).
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0   # ms

    # warmup (CUDA context, caches, clocks)
    run_h2d(); run_gemm(); torch.cuda.synchronize()

    # best-of-3 to reduce noise
    t_h2d = min(timed(run_h2d) for _ in range(3))
    t_gemm = min(timed(run_gemm) for _ in range(3))

    def both():
        run_h2d()      # enqueue on h2d_stream
        run_gemm()     # enqueue on exec_stream -> runs concurrently
    t_both = min(timed(both) for _ in range(3))

    serial = t_h2d + t_gemm
    overlap_ratio = 1.0 - (t_both / serial)
    print(f"\n[stream-overlap] h2d={t_h2d:.2f}ms gemm={t_gemm:.2f}ms "
          f"both={t_both:.2f}ms serial={serial:.2f}ms "
          f"overlap={overlap_ratio*100:.1f}%")

    # concurrent must be meaningfully below the serial sum => the two streams
    # progressed in parallel.  Margin is generous to avoid flakiness; true
    # overlap on H100 is typically 30-50%.
    assert t_both < serial * 0.85, (
        f"no fetch‖exec overlap: both={t_both:.2f}ms not < "
        f"0.85*serial={0.85*serial:.2f}ms (h2d={t_h2d:.2f} gemm={t_gemm:.2f})")


# ==================================================================
# 2b. HARDWARE — MISS pipeline: fetch[i] ‖ compute[i-1]
#     A miss's OWN exec must wait for its OWN fetch (data dependency, the
#     per-task cudaEventSynchronize).  The overlap that hides PCIe latency is
#     ACROSS experts: while expert i computes on exec_stream, expert i+1's
#     H2D fetch runs on h2d_stream.  This reproduces that pipeline and shows
#     total ≈ max(Σfetch, Σcompute), not Σfetch+Σcompute.
# ==================================================================
@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_hardware_miss_pipeline_overlap():
    import time

    import torch

    dev = torch.device("cuda:0")
    h2d_stream = torch.cuda.Stream(dev)
    exec_stream = torch.cuda.Stream(dev)

    N = 24                 # experts fetched+computed this "layer"
    n = 4096               # weight is n×n fp32 (64 MiB) ~ fetch time ~ compute
    hosts = [torch.empty(n, n, dtype=torch.float32, pin_memory=True)
             for _ in range(N)]          # host weights (PCIe source)
    devw = [torch.empty(n, n, device=dev) for _ in range(N)]  # slot buffers
    x = torch.randn(n, n, device=dev)
    evs = [torch.cuda.Event() for _ in range(N)]

    def fetch_only():
        with torch.cuda.stream(h2d_stream):
            for i in range(N):
                devw[i].copy_(hosts[i], non_blocking=True)

    def compute_only():
        with torch.cuda.stream(exec_stream):
            acc = x
            for i in range(N):
                acc = torch.mm(acc, devw[i])
        return acc

    def pipelined():
        # dispatcher pattern: all fetches race ahead on h2d_stream recording a
        # per-expert event; exec_stream computes expert i only after its OWN
        # fetch event (the miss data dependency), while fetch i+1.. overlap
        # compute i.
        with torch.cuda.stream(h2d_stream):
            for i in range(N):
                devw[i].copy_(hosts[i], non_blocking=True)
                evs[i].record(h2d_stream)
        with torch.cuda.stream(exec_stream):
            acc = x
            for i in range(N):
                exec_stream.wait_event(evs[i])      # gate on OWN fetch only
                acc = torch.mm(acc, devw[i])
        return acc

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    fetch_only(); compute_only(); pipelined()              # warmup
    torch.cuda.synchronize()
    t_fetch = min(timed(fetch_only) for _ in range(5))
    t_compute = min(timed(compute_only) for _ in range(5))
    t_pipe = min(timed(pipelined) for _ in range(5))

    serial = t_fetch + t_compute
    saved = 1.0 - t_pipe / serial
    print(f"\n[miss-pipeline] fetch_only={t_fetch:.2f}ms "
          f"compute_only={t_compute:.2f}ms (serial sum={serial:.2f}ms) "
          f"pipelined(fetch_i ‖ compute_(i-1))={t_pipe:.2f}ms saved={saved*100:.1f}%")

    # The gated pipeline (each compute waits its OWN fetch) must finish well
    # below fetch+compute run back-to-back => miss fetch overlapped exec.
    assert t_pipe < serial * 0.85, (
        f"miss fetch did not overlap exec: pipelined={t_pipe:.2f}ms not < "
        f"0.85*(fetch+compute)={0.85*serial:.2f}ms")


# ==================================================================
# 3. PRODUCTION METRIC — fetch_wait_us from get_phase_times()
# ==================================================================
def overlap_ratio_from_phase_times(phase_times, est_fetch_us):
    """phase_times = [fetch_wait_us, compute_us, combine_us] from the dispatcher.

    fetch_wait_us is the CPU time the exec worker blocked on fetch_event.  If
    fetch fully overlapped prior compute, fetch_wait_us ≪ est_fetch_us (the raw
    H2D time).  Returns 1 - fetch_wait/est_fetch (1.0 = perfect overlap).
    """
    fetch_wait = float(phase_times[0])
    if est_fetch_us <= 0:
        return 0.0
    return max(0.0, 1.0 - fetch_wait / est_fetch_us)


def _real_gate():
    if os.environ.get("MOE_EP_RUN_REAL_ARCHER_TEST") != "1":
        return "set MOE_EP_RUN_REAL_ARCHER_TEST=1 to run the real dispatcher"
    try:
        from moe_infinity import _store  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"archer extension not importable: {e}"
    return ("real fetch‖exec overlap needs a loaded model (register_expert "
            "+ real expert weights); read get_phase_times()[0]=fetch_wait_us "
            "from a benchmark run (scripts/m2_forward.py) and assert "
            "overlap_ratio_from_phase_times(pt, est_fetch_us) > 0.3")


@pytest.mark.skipif(_real_gate() is not None, reason=str(_real_gate()))
def test_real_phase_times_overlap():  # pragma: no cover - needs model harness
    pytest.skip("documented: drive a real layer, then assert fetch_wait_us "
                "<< H2D time via overlap_ratio_from_phase_times")


def test_overlap_helper_math():
    # pure-python sanity for the metric helper
    assert overlap_ratio_from_phase_times([0, 100, 5], 1000) == 1.0
    assert overlap_ratio_from_phase_times([1000, 100, 5], 1000) == 0.0
    assert abs(overlap_ratio_from_phase_times([300, 0, 0], 1000) - 0.7) < 1e-9


if __name__ == "__main__":
    test_struct_streams_distinct_and_hits_nonblocking()
    test_overlap_helper_math()
    if _has_cuda():
        test_hardware_h2d_gemm_overlap()
    print("stream overlap: ALL PASS")
