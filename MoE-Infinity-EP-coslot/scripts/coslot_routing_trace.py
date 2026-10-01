"""Routing/plan/cache trace capture — batch=4 per rank, EP=4, configurable cap.

For each rank: a batch of 4 distinct prompts (= 4 requests), greedy generate
``--max_new_tokens`` decode steps.  With MOE_EP_ROUTING_TRACE=1 the executor
dumps one JSONL line per layer containing the global Controller plan
(hit_ops + fetch_ops), the unified expert→GPU routing, per-expert demand, and
the Archer cache slot state before/after (= shadow, drift==0 vs archer).

Each rank also writes a request manifest (request_id → prompt → token count).

Run via scripts (sets cap=4 config + NUMA-shared + trace env), e.g.:
  MOE_EP_ROUTING_TRACE=1 MOE_EP_ROUTING_TRACE_PATH=<dir> MOE_EP_RUN_TAG=trace \
  torchrun ... numa_wrap.sh coslot_routing_trace.py <cfg> --max_new_tokens 5
"""
from __future__ import annotations

import argparse
import json
import os
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
BATCH_PER_RANK = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--run_tag", default="trace")
    args = parser.parse_args()

    # Ensure the executor sees the trace flags (read in EPExpertExecutor.__init__
    # during engine.load()).  RUN_TAG names the per-rank JSONL files.
    os.environ["MOE_EP_ROUTING_TRACE"] = "1"
    os.environ.setdefault("MOE_EP_ROUTING_TRACE_PATH", "/tmp/moe_ep_traces")
    os.environ["MOE_EP_RUN_TAG"] = args.run_tag
    trace_dir = os.environ["MOE_EP_ROUTING_TRACE_PATH"]
    os.makedirs(trace_dir, exist_ok=True)

    cfg = Config.from_yaml(args.config)
    world_size = int(os.environ["WORLD_SIZE"])
    cfg.resolve_parallel(world_size)
    topology = init_distributed(cfg.parallel.ep_size)

    rank = topology.global_rank
    # per-rank batch of 4 distinct requests (wrap around the prompt list).
    my_prompts = [PROMPTS[(rank + i) % len(PROMPTS)] for i in range(BATCH_PER_RANK)]
    request_ids = [f"req_r{rank}_b{i}" for i in range(BATCH_PER_RANK)]

    if topology.is_rank0:
        print(f"[setup] world={topology.world_size} ep={topology.ep_size} "
              f"cap={cfg.offload.cache_capacity_per_rank or 'unlimited'} "
              f"batch_per_rank={BATCH_PER_RANK} decode_steps={args.max_new_tokens}",
              flush=True)

    t0 = time.time()
    engine = MoE_EP(cfg, topology)
    engine.load()
    if topology.is_rank0:
        print(f"[load] took {time.time()-t0:.1f}s", flush=True)

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batched decode
    enc = tokenizer(my_prompts, return_tensors="pt", padding=True)
    input_ids = enc.input_ids.cuda()
    attention_mask = enc.attention_mask.cuda()

    # per-rank request manifest.
    manifest = {
        "rank": rank,
        "gpu": rank,
        "batch_size": BATCH_PER_RANK,
        "decode_steps": args.max_new_tokens,
        "requests": [
            {"request_id": request_ids[i], "prompt": my_prompts[i],
             "prompt_tokens": int(attention_mask[i].sum().item())}
            for i in range(BATCH_PER_RANK)
        ],
    }
    with open(f"{trace_dir}/{args.run_tag}_requests_rank{rank}.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    model = engine.model
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    decoded = [tokenizer.decode(gen[i], skip_special_tokens=True)
               for i in range(gen.size(0))]
    print(f"[rank{rank}] generate {dt:.2f}s decoded[0]={decoded[0][:80]!r}",
          flush=True)

    barrier()
    if topology.is_rank0:
        print("[coslot-trace] OK", flush=True)
    torch.cuda.synchronize()
    import torch.distributed as dist
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    ilog.proc_mem("pre_exit(normal)")
    os._exit(0)


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = int(main() or 0)
    except BaseException as _e:
        import traceback
        traceback.print_exc()
        print(f"[coslot-trace] FATAL: {_e!r}", flush=True)
        _rc = 1
    finally:
        ilog.proc_mem(f"pre_exit(rc={_rc})")
        os._exit(_rc)
