#!/usr/bin/env python3
"""Real CUDA/NCCL slot execution against an independent PyTorch oracle."""
import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from mgo_v2.bootstrap import pin_rank_before_cuda_import

BOOT = pin_rank_before_cuda_import()
os.environ.setdefault("MOE_INFINITY_DISABLE_MLOCK", "1")
os.environ.setdefault("MOE_INFINITY_MAX_TOKENS", "128")
os.environ.setdefault("MOE_IO_THREADS", "2")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.executor import LegacySlotExecutorAdapter
from mgo_v2.native import load_slot_extension, register_experts
from mgo_v2.runtime import DistributedMoERuntime
from mgo_v2.affinity import AffinityTables
from mgo_v2.profiling import CollectiveStats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--events", type=int, default=12)
    p.add_argument("--cache-ratio", type=float, default=1.0)
    p.add_argument("--eviction", choices=["lru", "gate", "coverage"], default="lru")
    p.add_argument("--admission", default="random")
    p.add_argument("--substitution", action="store_true")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--intermediate-size", type=int, default=64)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--reset-at", type=int, help="reset residency and resize the pool midway through the fixture")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    dist.all_reduce(torch.ones(1, device="cuda"))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261001)
    layers, experts, hidden, intermediate, top_k = 3, 16, args.hidden_size, args.intermediate_size, 4
    host = {(l, e): [torch.randn(shape, dtype=torch.bfloat16) * 0.05
                     for shape in [(intermediate, hidden), (intermediate, hidden), (hidden, intermediate)]]
            for l in range(layers) for e in range(experts)}
    lib = load_slot_extension()
    handle, dispatcher = register_experts(lib, root / f"weights-rank{rank}",
        ((l, e, tensors) for (l, e), tensors in host.items()), layers, experts)
    config = RuntimeConfig(num_layers=layers, num_experts=experts, top_k=top_k,
                           world_size=world, global_cache_ratio=args.cache_ratio,
                           admission=args.admission, eviction=args.eviction,
                           substitution_enabled=args.substitution)
    sim = np.tile(np.eye(experts, dtype=np.float32), (layers, 1, 1))
    if args.substitution:
        for base in range(0, experts, 4):
            sim[:, base + 2, base] = .8
            sim[:, base + 3, base + 1] = .8
    rng = np.random.default_rng(42)
    affinity = AffinityTables(rng.random((layers, experts, experts)), rng.random((layers - 1, experts, experts)))
    ctrl = GlobalExpertController(config, sim, affinity)
    executor = LegacySlotExecutorAdapter(dispatcher, config.per_rank_slots()[rank], 0, experts)
    observed = {}
    def observe(routes, plan, output):
        observed["routes"] = [r for origin, r in zip(routes.origin_ranks, plan.effective_token_routes) if origin == rank]
    stats = CollectiveStats() if args.profile else None
    runtime = DistributedMoERuntime(ctrl, executor, debug=True, observer=observe, collective_stats=stats)
    max_error = 0.0
    rows = []
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    with torch.inference_mode():
        for step in range(args.events):
            if step == args.reset_at:
                config = replace(config, global_cache_ratio=.75)
                dispatcher.reset_slot_pool(config.per_rank_slots()[rank])
                assert dispatcher.get_cached_slots(0) == []
                ctrl = GlobalExpertController(config, sim, affinity)
                executor = LegacySlotExecutorAdapter(dispatcher, config.per_rank_slots()[rank], 0, experts)
                runtime = DistributedMoERuntime(ctrl, executor, debug=True, observer=observe, collective_stats=stats)
            layer = step % layers
            torch.manual_seed(1000 + step * world + rank)
            # Include an empty source rank and an all-empty event.
            n = 0 if step == args.events - 1 or (rank == world - 1 and world > 1) else 7 + rank
            x = torch.randn((n, hidden), dtype=torch.bfloat16, device="cuda")
            probs = torch.randn((x.shape[0], experts), device="cuda").softmax(-1)
            weights, ids = probs.topk(top_k, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
            if args.substitution:
                base = ((step // layers) * 4) % experts
                ids = torch.arange(base, base + 4, device="cuda").repeat(n, 1)
                weights = x.new_tensor([.55, .25, .12, .08]).repeat(n, 1)
                probs = torch.zeros((n, experts), device="cuda").scatter_(1, ids, weights.float())
            with torch.cuda.nvtx.range("mgo_runtime"):
                if step % 2:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual = runtime.forward_layer(layer, x, ids, weights, probs, False)
                    torch.cuda.current_stream().wait_stream(stream)
                else:
                    actual = runtime.forward_layer(layer, x, ids, weights, probs, False)
            expected = torch.zeros_like(x)
            effective_weights = torch.zeros((n, experts), device="cuda", dtype=x.dtype)
            for token, routed in enumerate(observed["routes"]):
                for expert, weight in routed.items():
                    effective_weights[token, expert] = weight
            for e in range(experts):
                token = effective_weights[:, e].nonzero().flatten()
                if token.numel():
                    gate, up, down = [w.cuda() for w in host[layer, e]]
                    y = F.linear(F.silu(F.linear(x[token], gate)) * F.linear(x[token], up), down)
                    expected.index_add_(0, token, (y * effective_weights[token, e, None]).to(x.dtype))
            torch.cuda.synchronize()
            err = (actual.float() - expected.float()).abs().max().item() if n else 0.0
            max_error = max(max_error, err)
            physical = set(map(tuple, dispatcher.get_cached_experts(0)))
            assert physical == set(ctrl.cache.keys_on_rank(rank)), (step, physical, ctrl.cache.owner)
            rows.append({"step": step, "max_abs_error": err, "finite": bool(actual.isfinite().all())})
            print(json.dumps({"rank": rank, **rows[-1]}), flush=True)
            torch.testing.assert_close(actual, expected,
                                       atol=0.0 if executor.native_numerics else 0.006,
                                       rtol=0.0 if executor.native_numerics else 0.06)
    result = {"status": "PASS", "native_numerics": executor.native_numerics,
              "slot_views": os.environ.get("MOE_EP_SLOT_VIEWS", "0") == "1",
              "rank": rank, "world": world, "boot": BOOT, "max_abs_error": max_error,
              "events": rows, "cache_stats": dispatcher.get_cache_stats().tolist(),
              "metrics": runtime.metrics.to_dict(), "eviction": args.eviction, "admission": args.admission,
              "controller_seconds": runtime.controller_seconds}
    result["reset_at"] = args.reset_at
    if args.profile:
        result["collectives"] = stats.summary()
        torch.cuda.cudart().cudaProfilerStop()
    if args.substitution:
        assert runtime.metrics.subhit > 0
    (root / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    del runtime, executor, dispatcher
    handle.clean_up_resources()


if __name__ == "__main__":
    main()
