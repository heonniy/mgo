"""Milestone-3 smoke: multi-rank unified EP forward (NPROC>=2).

Each rank shards the prompt batch (batch dim sharded across world). Forward
exercises the full M3 path: router → cache view all-gather → policy plan →
NVLink token routing → local expert exec → all-to-all back → combine.

For naive policies + unified EP, with a single prompt replicated across ranks
(batch=1 per rank), the output of each rank should match the M2 single-rank
result for the same prompt.
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
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    world_size = int(os.environ["WORLD_SIZE"])
    cfg.resolve_parallel(world_size)

    topology = init_distributed(cfg.parallel.ep_size)
    if topology.is_rank0:
        print(f"[setup] world={topology.world_size} ep={topology.ep_size}", flush=True)

    t0 = time.time()
    engine = MoE_EP(cfg, topology)
    engine.load()
    if topology.is_rank0:
        print(f"[load] took {time.time()-t0:.1f}s", flush=True)

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)

    # Each rank gets its OWN local batch shard. For this smoke we just use
    # the same prompt on each rank (so all ranks produce the same forward
    # result, which we can sanity-check against the M2 single-rank output).
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()
    if topology.is_rank0:
        print(f"[input] prompt={args.prompt!r} ids={input_ids.tolist()}", flush=True)

    model = engine.model
    model.eval()

    # Forward — exercises multi-rank MoE blocks.
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids)
    dt = time.time() - t0
    logits = out.logits

    # Rank-local snapshot.
    last = logits[0, -1].float().cpu()
    argmax_last = int(last.argmax())
    top5_last = torch.topk(last, 5).indices.tolist()
    top5_logits = torch.topk(last, 5).values.tolist()
    # Bytewise hash of the entire last-token logit row (FP32) — two ranks
    # with identical inputs should produce the same hash if the multi-rank
    # path is numerically deterministic across ranks.
    import hashlib
    logit_hash = hashlib.sha1(last.numpy().tobytes()).hexdigest()[:16]
    print(
        f"[rank{topology.global_rank}] forward {dt:.2f}s "
        f"argmax={argmax_last} ({tokenizer.decode([argmax_last])!r}) "
        f"top5={top5_last} "
        f"top5_logits={[round(v, 4) for v in top5_logits]} "
        f"hash={logit_hash} "
        f"sum={last.sum().item():.4f} max={last.max().item():.4f}",
        flush=True,
    )

    # Greedy generate.
    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    print(
        f"[rank{topology.global_rank}] generate {dt:.2f}s "
        f"decoded={tokenizer.decode(gen_out[0])!r}",
        flush=True,
    )

    barrier()
    if topology.is_rank0:
        print("[M3] forward OK", flush=True)

    import torch.distributed as dist
    torch.cuda.synchronize()
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[M3] destroy_process_group: {e!r}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = int(main() or 0)
    except BaseException as _e:
        import traceback
        traceback.print_exc()
        print(f"[M3] FATAL exception in main: {_e!r}", flush=True)
        _rc = 1
    finally:
        os._exit(_rc)
