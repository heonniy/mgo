#!/usr/bin/env python3
"""Minimal R1/R4/R8 model-execution smoke for mgo_v2.

This is a correctness harness, not a paper benchmark.
"""

import argparse
import os

from mgo_v2.bootstrap import pin_rank_before_cuda_import

BOOT = pin_rank_before_cuda_import()

# CUDA libraries may be imported only after visibility is pinned.
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from moe_infinity import MoE

from mgo_v2.affinity import AffinityTables
from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.executor import LegacySlotExecutorAdapter
from mgo_v2.qwen3_integration import attach_qwen3_runtime
from mgo_v2.runtime import DistributedMoERuntime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--offload-dir", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument("--affinity", default=None)
    p.add_argument("--cache-ratio", type=float, default=0.30)
    p.add_argument(
        "--eviction", choices=["lru", "gate", "coverage"], default="coverage"
    )
    p.add_argument(
        "--admission",
        choices=[
            "random",
            "greedy_current",
            "greedy_path",
            "hungarian_current",
            "hungarian_same",
            "hungarian_same_path",
            "hungarian_swap",
        ],
        default="hungarian_same_path",
    )
    p.add_argument("--prompt", default="What is 17 + 25?")
    p.add_argument("--max-new-tokens", type=int, default=4)
    args = p.parse_args()

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")
    rank, world = dist.get_rank(), dist.get_world_size()

    similarity = np.load(args.similarity)
    affinity = AffinityTables.load_npz(args.affinity) if args.affinity else None

    config = RuntimeConfig(
        num_layers=int(similarity.shape[0]),
        num_experts=int(similarity.shape[1]),
        global_cache_ratio=args.cache_ratio,
        world_size=world,
        eviction=args.eviction,
        admission=args.admission,
    )

    moe = MoE(
        args.model,
        {
            "offload_path": os.path.join(args.offload_dir, f"rank{rank}"),
            "device_memory_ratio": 0.90,
            "prefetch": False,
        },
    )

    controller = GlobalExpertController(config, similarity, affinity)
    executor = LegacySlotExecutorAdapter(
        moe.engine.expert_dispatcher,
        capacity=config.per_rank_slots()[rank],
        expert_bytes=0,  # legacy slot executor auto-detects registered expert bytes
        num_experts=config.num_experts,
    )
    runtime = DistributedMoERuntime(controller, executor)
    patched = attach_qwen3_runtime(moe.model, runtime)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok([args.prompt], return_tensors="pt", padding=True)
    enc = {k: v.to("cuda:0") for k, v in enc.items()}

    with torch.no_grad():
        out = moe.model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(
        {
            "rank": rank,
            "world": world,
            "boot": BOOT,
            "patched_blocks": patched,
            "output": text if rank == 0 else "<suppressed>",
        },
        flush=True,
    )
    dist.barrier()


if __name__ == "__main__":
    main()
