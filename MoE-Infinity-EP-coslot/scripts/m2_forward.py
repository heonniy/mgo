"""Milestone-2 smoke: single-rank forward through our DistributedOffloadEngine.

Loads the model, runs a tiny prompt through the FULL forward path (including
our Qwen3MoEBlockEP and EPExpertExecutor.run_layer), prints the first row of
logits. Validates end-to-end integration: block swap, executor wiring, expert
dispatcher reuse, archer fetch on demand.

Strictly NPROC=1 (single rank, ep_size==1). NPROC>1 is Milestone 3.

Usage:
    bash scripts/run_m2.sh [config.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from moe_infinity_ep.launch.distributed_setup import pin_visible_device

pin_visible_device()

from moe_infinity_ep.launch.distributed_setup import init_distributed, barrier  # noqa: E402
from moe_infinity_ep.launch.entry import MoE_EP  # noqa: E402
from moe_infinity_ep.utils.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument(
        "--save_ref", default=None,
        help="Directory to write a reference JSON (last_logits_fp32) for this "
             "prompt. Used by scripts/m3_verify.py.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 1:
        raise SystemExit(
            f"M2 verification expects WORLD_SIZE=1, got {world_size}. "
            "Multi-rank path arrives in Milestone 3."
        )
    cfg.resolve_parallel(world_size)

    topology = init_distributed(cfg.parallel.ep_size)
    if topology.is_rank0:
        print(f"[setup] world={topology.world_size} ep={topology.ep_size}", flush=True)

    t0 = time.time()
    engine = MoE_EP(cfg, topology)
    engine.load()
    print(f"[load] took {time.time()-t0:.1f}s", flush=True)

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()
    print(f"[input] prompt={args.prompt!r} ids={input_ids.tolist()}", flush=True)

    model = engine.model
    model.eval()

    # Forward pass — exercises the MoE blocks.
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids)
    dt = time.time() - t0
    logits = out.logits  # [1, T, vocab]
    print(f"[forward] took {dt:.2f}s, logits.shape={tuple(logits.shape)}", flush=True)
    last_fp32 = logits[0, -1].float().cpu()
    argmax_id = int(last_fp32.argmax())
    top5 = torch.topk(last_fp32, 5).indices.tolist()
    print(f"[forward] last-token argmax: {argmax_id} "
          f"(decoded: {tokenizer.decode([argmax_id])!r})", flush=True)
    print(f"[forward] last-token top5: {top5}", flush=True)

    if args.save_ref:
        import hashlib, json
        from pathlib import Path
        ref_dir = Path(args.save_ref)
        ref_dir.mkdir(parents=True, exist_ok=True)
        slug = hashlib.sha1(args.prompt.encode("utf-8")).hexdigest()[:12]
        ref = {
            "prompt": args.prompt,
            "input_ids": input_ids.cpu().tolist(),
            "argmax": argmax_id,
            "top5": top5,
            "last_logits_fp32": last_fp32.tolist(),
        }
        ref_path = ref_dir / f"{slug}.json"
        ref_path.write_text(json.dumps(ref))
        print(f"[ref] saved {ref_path}", flush=True)

    # Greedy generate a few tokens — exercises the decode path (sequence_length==1).
    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    print(f"[generate] took {dt:.2f}s, "
          f"new_tokens={args.max_new_tokens}, "
          f"decoded={tokenizer.decode(gen_out[0])!r}", flush=True)

    barrier()
    print("[M2] forward OK", flush=True)

    # See note in m1_smoke.py — hard-exit to bypass archer's long-lived C++
    # threads that hang Python shutdown.
    import torch
    import torch.distributed as dist
    torch.cuda.synchronize()
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[M2] destroy_process_group: {e!r}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = int(main() or 0)
    except BaseException as _e:
        import traceback
        traceback.print_exc()
        print(f"[M2] FATAL exception in main: {_e!r}", flush=True)
        _rc = 1
    finally:
        os._exit(_rc)
