"""Milestone-1 smoke test.

Each rank pins its GPU, initializes NCCL EP/DP groups, loads the configured
model via DistributedOffloadEngine, prints a per-rank report, barriers, and
exits cleanly. No forward pass.

Usage (via scripts/run_smoke.sh, which sets RANK/WORLD_SIZE/LOCAL_RANK via torchrun):

    torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \\
        scripts/m1_smoke.py configs/qwen3_235b_auto.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# CRITICAL: pin the GPU BEFORE importing anything that may probe CUDA
# (moe_infinity's C++ extension does this at import time). See plan §"kNumDevices timing".
from moe_infinity_ep.launch.distributed_setup import pin_visible_device

pin_visible_device()

from moe_infinity_ep.launch.distributed_setup import init_distributed, barrier  # noqa: E402
from moe_infinity_ep.launch.entry import MoE_EP  # noqa: E402
from moe_infinity_ep.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config (e.g., configs/qwen3_235b_auto.yaml)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    world_size = int(os.environ["WORLD_SIZE"])
    cfg.resolve_parallel(world_size)

    t0 = time.time()
    topology = init_distributed(cfg.parallel.ep_size)
    t_dist = time.time() - t0

    if topology.is_rank0:
        print(
            f"[setup] world={topology.world_size} ep={topology.ep_size} "
            f"(unified; each rank has own batch shard) "
            f"(init: {t_dist*1000:.0f} ms)",
            flush=True,
        )

    t0 = time.time()
    engine = MoE_EP(cfg, topology)
    engine.load()
    t_load = time.time() - t0

    rep = engine.report()
    rep["load_seconds"] = round(t_load, 2)

    # Each rank prints its own line so we can confirm all came up
    print(f"[rank {topology.global_rank}] {json.dumps(rep)}", flush=True)

    barrier()
    if topology.is_rank0:
        print("[M1] smoke OK — all ranks loaded model and reached final barrier", flush=True)

    # The archer C++ extension spawns ~30 long-lived AIO threads + worker
    # threads per rank; Python's normal shutdown waits forever for those.
    # Best-effort tear down NCCL, then hard-exit so the smoke actually returns.
    import torch
    import torch.distributed as dist
    torch.cuda.synchronize()
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[M1] destroy_process_group: {e!r}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    # M13 fix (2026-05-27): wrap in try/finally so any exception path also
    # routes to os._exit. Without this, an exception in main() unwinds
    # the stack normally → archer C++ destructors run → 30 AIO threads hang
    # → torchrun has to SIGKILL → process leak.
    _rc = 1
    try:
        _rc = int(main() or 0)
    except BaseException as _e:
        import traceback
        traceback.print_exc()
        print(f"[M1] FATAL exception in main: {_e!r}", flush=True)
        _rc = 1
    finally:
        os._exit(_rc)
