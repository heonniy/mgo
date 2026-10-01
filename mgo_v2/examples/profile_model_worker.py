#!/usr/bin/env python3
"""Nsight worker wrapper; run under torchrun with a fresh benchmark output.

Profiling starts after loading and the two-token warmup. The normal benchmark
is reused unchanged. These timings are diagnostic, never matrix speed claims.
"""
import argparse
import os
import faulthandler
import signal
import hashlib
import json
from pathlib import Path

# This import pins rank visibility before importing any CUDA library.
import benchmark_model as benchmark


def main():
    # Opt-in, process-local native stack inspection for profiler stalls.
    if os.environ.get("MGO_DEBUG_PTRACE") == "1":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(0x59616D61, ctypes.c_ulong(-1).value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_PTRACER failed")
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.enable()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args, _ = parser.parse_known_args()
    root = Path(args.output)
    if list(root.glob("*-rep*-rank*.json")):
        raise ValueError("Nsight profiling requires a fresh output directory")
    labels = [(cell["name"], repeat) for cell in json.loads(Path(args.cells).read_text())
              for repeat in range(args.repeats)]
    original = benchmark.generate
    calls, measured, started = 0, 0, False
    def traced(*values, **kwargs):
        nonlocal calls, measured, started
        calls += 1
        if calls == 1:  # benchmark_model's only unmeasured warmup
            return original(*values, **kwargs)
        if not started:
            stack_seconds = int(os.environ.get("MGO_STACK_DUMP_SECONDS", "0"))
            if stack_seconds > 0:
                faulthandler.dump_traceback_later(stack_seconds, repeat=True)
            benchmark.torch.cuda.cudart().cudaProfilerStart()
            started = True
        name, repeat = labels[measured]
        rank = benchmark.dist.get_rank()
        measured += 1
        label = f"mgo_cell:{rank}:{name}:{repeat}"
        with benchmark.torch.cuda.nvtx.range(label):
            result = original(*values, **kwargs)
            benchmark.torch.cuda.synchronize()
        return result
    benchmark.generate = traced
    try:
        benchmark.main()
    finally:
        if started:
            benchmark.torch.cuda.cudart().cudaProfilerStop()
    faulthandler.cancel_dump_traceback_later()
    if measured != len(labels):
        raise AssertionError("profiling did not measure every requested cell")
    receipt = {"rank": benchmark.BOOT["local_rank"], "measured_cells": measured,
               "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Nsight diagnostic; excludes loading, includes profiled fixed-step generation."}
    (root / f"nsight-rank{receipt['rank']}.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
