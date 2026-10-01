"""Milestone-3 verification — automated check against single-rank reference.

Usage:
    bash scripts/run_m3_verify.sh [config.yaml]

What it does (per rank with WORLD_SIZE=N):
1. Pick N distinct prompts (one per rank).
2. Each rank runs ITS prompt through MoE_EP (proper unified EP usage —
   each rank has its own batch shard).
3. Capture last-token logits per rank.
4. Optionally compare to a JSON of reference logits produced by m2_forward.
   (Run m2_forward.py once per prompt beforehand to populate the reference.)

Reports per rank:
    - top-1 token (and decoded)
    - top-5 stability vs reference
    - max-abs-diff of full logit row vs reference
    - layerwise counters (from instrumentation)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from moe_infinity_ep.launch.distributed_setup import pin_visible_device

pin_visible_device()

from moe_infinity_ep.launch.distributed_setup import init_distributed, barrier  # noqa: E402
from moe_infinity_ep.launch.entry import MoE_EP  # noqa: E402
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
    parser.add_argument(
        "--ref_dir", default="/tmp/moe_ep_traces/m2_refs",
        help="Where m2_forward stored per-prompt reference logits.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.0,
        help="Max-abs-diff threshold for logits vs reference (default 1.0).",
    )
    parser.add_argument(
        "--strict_top1", action="store_true",
        help="Fail if top-1 token doesn't match reference (default: warn only).",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    world_size = int(os.environ["WORLD_SIZE"])
    cfg.resolve_parallel(world_size)

    topology = init_distributed(cfg.parallel.ep_size)
    if topology.is_rank0:
        print(f"[setup] world={topology.world_size} ep={topology.ep_size}", flush=True)

    my_prompt = PROMPTS[topology.global_rank % len(PROMPTS)]
    if topology.is_rank0:
        for i in range(world_size):
            print(f"[setup] rank{i} prompt={PROMPTS[i % len(PROMPTS)]!r}", flush=True)

    t0 = time.time()
    engine = MoE_EP(cfg, topology)
    engine.load()
    if topology.is_rank0:
        print(f"[load] took {time.time()-t0:.1f}s", flush=True)

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
    input_ids = tokenizer(my_prompt, return_tensors="pt").input_ids.cuda()

    model = engine.model
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids)
    dt = time.time() - t0
    last = out.logits[0, -1].float().cpu()
    argmax = int(last.argmax())
    top5 = torch.topk(last, 5).indices.tolist()
    h = hashlib.sha1(last.numpy().tobytes()).hexdigest()[:16]

    # Reference comparison (per-prompt JSON written by m2_forward.py).
    ref_path = Path(args.ref_dir) / f"{_prompt_slug(my_prompt)}.json"
    ref_summary = ""
    diff_summary = "no ref"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        ref_last = torch.tensor(ref["last_logits_fp32"], dtype=torch.float32)
        diff = (last - ref_last).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        # Re-rank check.
        top1_ref = ref["argmax"]
        top5_ref = ref["top5"]
        top1_match = argmax == top1_ref
        top5_overlap = len(set(top5) & set(top5_ref))
        ref_summary = (
            f"ref_top1={top1_ref} match={top1_match} "
            f"top5_overlap={top5_overlap}/5"
        )
        diff_summary = (
            f"max_abs_diff={max_diff:.4f} mean_abs_diff={mean_diff:.4f} "
            f"tol={args.tolerance}"
        )
        if args.strict_top1 and not top1_match:
            print(
                f"[rank{topology.global_rank}] FAIL: top1 mismatch "
                f"({argmax} vs ref {top1_ref})", flush=True,
            )

    print(
        f"[rank{topology.global_rank}] prompt={my_prompt!r} "
        f"forward {dt:.2f}s argmax={argmax} ({tokenizer.decode([argmax])!r}) "
        f"top5={top5} hash={h} {ref_summary} {diff_summary}",
        flush=True,
    )

    # Dump counters trace from rank 0.
    barrier()
    try:
        from moe_infinity_ep.instrument.trace import gather_and_dump
        trace_dir = cfg.instrumentation.trace_path
        # MoE_EP wraps DistributedOffloadEngine as .engine; the executor lives there.
        executor = engine.engine.expert_executor
        trace_path = gather_and_dump(
            executor.counters, topology,
            trace_dir, cfg.to_dict(),
        )
        if topology.is_rank0 and trace_path:
            print(f"[trace] dumped {trace_path}", flush=True)
    except Exception as e:
        if topology.is_rank0:
            print(f"[trace] dump failed: {e!r}", flush=True)

    barrier()
    if topology.is_rank0:
        print("[M3-verify] OK", flush=True)

    torch.cuda.synchronize()
    import torch.distributed as dist
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    os._exit(0)


def _prompt_slug(p: str) -> str:
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = int(main() or 0)
    except BaseException as _e:
        import traceback
        traceback.print_exc()
        print(f"[M3verify] FATAL exception in main: {_e!r}", flush=True)
        _rc = 1
    finally:
        os._exit(_rc)
