"""Long greedy generate to exercise cache hit/miss/eviction over many layers.

Each rank generates ``--max_new_tokens`` tokens (default 32) on its own prompt.
Per-layer Counters accumulate, then trace is dumped via gather_and_dump.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

from moe_infinity_ep.launch.distributed_setup import pin_visible_device

pin_visible_device()

from moe_infinity_ep.launch.distributed_setup import init_distributed, barrier  # noqa: E402
from moe_infinity_ep.launch.entry import MoE_EP  # noqa: E402
from moe_infinity_ep.launch import init_logging as ilog  # noqa: E402
from moe_infinity_ep.utils.config import Config  # noqa: E402


PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Photosynthesis converts",
    "The author of 'Pride and Prejudice' is",
    "The chemical symbol for gold is",
    "The speed of light is approximately",
    "The currency of Japan is",
    "The longest river in the world is",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--run_tag", default="long_gen")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    world_size = int(os.environ["WORLD_SIZE"])
    cfg.resolve_parallel(world_size)

    topology = init_distributed(cfg.parallel.ep_size)
    if topology.is_rank0:
        print(
            f"[setup] world={topology.world_size} ep={topology.ep_size} "
            f"fetch_policy={cfg.policies.fetch_dispatch} "
            f"placement={cfg.policies.placement} "
            f"cache_capacity={cfg.offload.cache_capacity_per_rank or 'unlimited'} "
            f"prefetch={cfg.execution.enable_prefetch}",
            flush=True,
        )

    my_prompt = PROMPTS[topology.global_rank % len(PROMPTS)]
    if topology.is_rank0:
        for i in range(world_size):
            print(f"[setup] rank{i} prompt={PROMPTS[i % len(PROMPTS)]!r}", flush=True)

    t0 = time.time()
    engine = MoE_EP(cfg, topology)
    engine.load()
    if topology.is_rank0:
        print(f"[load] took {time.time()-t0:.1f}s", flush=True)

    # DEBUG: dump archer's cached_experts_ at end of init, BEFORE any forward.
    # Helps locate "extra_in_archer at L0 verify" — if cached_experts_ is
    # non-empty here, something during model load populated it.
    try:
        _exec = engine.engine.expert_executor
        _ld = _exec.local_dispatcher
        if _ld is not None and hasattr(_ld, "get_cached_experts"):
            _ce = _ld.get_cached_experts(0)
            print(f"[debug rank={topology.global_rank}] post-load "
                  f"cached_experts(gpu0) count={len(_ce)} "
                  f"sample={sorted(_ce)[:20]}", flush=True)
    except Exception as _e:
        print(f"[debug] post-load cache dump failed: {_e!r}", flush=True)

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
    input_ids = tokenizer(my_prompt, return_tensors="pt").input_ids.cuda()

    model = engine.model
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    decoded = tokenizer.decode(gen[0])
    print(
        f"[rank{topology.global_rank}] generate {dt:.2f}s "
        f"({args.max_new_tokens} new toks → {dt/args.max_new_tokens*1000:.0f} ms/tok) "
        f"decoded={decoded[:120]!r}",
        flush=True,
    )

    barrier()
    try:
        from moe_infinity_ep.instrument.trace import gather_and_dump
        run_id = f"{args.run_tag}_{cfg.policies.fetch_dispatch}_pf{int(cfg.execution.enable_prefetch)}_{time.strftime('%H%M%S')}"
        path = gather_and_dump(
            engine.engine.expert_executor.counters,
            topology, cfg.instrumentation.trace_path,
            cfg.to_dict(), run_id=run_id,
        )
        if topology.is_rank0 and path:
            print(f"[trace] dumped {path}", flush=True)
    except Exception as e:
        if topology.is_rank0:
            print(f"[trace] dump failed: {e!r}", flush=True)

    barrier()
    if topology.is_rank0:
        print("[M4-long-gen] OK", flush=True)

    torch.cuda.synchronize()
    import torch.distributed as dist
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    # Residue diagnosis: record this rank's locked/pinned/resident memory at
    # the moment of exit. os._exit(0) below skips atexit (NumaSharedRegion.close
    # never runs), so this is the only graceful snapshot of what we hand back
    # to the kernel to reclaim.
    ilog.proc_mem("pre_exit(normal)")
    os._exit(0)


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = int(main() or 0)
    except BaseException as _e:
        import traceback
        traceback.print_exc()
        print(f"[M4] FATAL exception in main: {_e!r}", flush=True)
        _rc = 1
    finally:
        ilog.proc_mem(f"pre_exit(rc={_rc})")
        os._exit(_rc)
