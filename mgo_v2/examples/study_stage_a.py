#!/usr/bin/env python3
"""Replay real Qwen3 decode events through resident slots and NCCL."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import benchmark_model as benchmark  # pins visibility before CUDA import
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from locality_maps import accounting, make_maps
from mgo_v2.affinity import AffinityTables
from mgo_v2.communicator import warmup_collectives, dispatch_tokens, return_partials
from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.executor import LegacySlotExecutorAdapter
from mgo_v2.model_loader import load_qwen3_slots
from mgo_v2.profiling import CollectiveStats
from mgo_v2.qwen3_integration import attach_qwen3_runtime, detach_qwen3_runtime
from mgo_v2.runtime import DistributedMoERuntime
from mgo_v2.types import AdmissionResult


class FixedAdmission:
    def __init__(self, owners, quotas):
        self.owners, self.quotas = owners, quotas

    def place(self, context):
        if not context.incoming:
            return AdmissionResult({}, [0] * context.world_size, "resident")
        assert set(context.incoming) == set(self.owners)
        return AdmissionResult(self.owners, self.quotas, "fixed_locality_map")


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "offload-dir", "similarity", "affinity", "workload", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--iterations", type=int, default=20)
    args = p.parse_args()
    assert args.iterations >= 20
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    warmup_collectives()
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if list(root.glob("rank*.json")):
        raise ValueError("Stage A requires fresh output")
    similarity = np.load(args.similarity)
    affinity = AffinityTables.load_npz(args.affinity)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = json.loads(Path(args.workload).read_text())[rank * 8:(rank + 1) * 8]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": row["question"] +
               "\nReturn only the final numeric answer, without explanation."}],
               tokenize=False, add_generation_prompt=True) for row in rows]
    encoded = {k: v.cuda() for k, v in tokenizer(prompts, padding=True, return_tensors="pt").items()}
    model, handle, dispatcher, store = load_qwen3_slots(args.model, args.offload_dir)
    config = RuntimeConfig(world_size=world, admission="random", eviction="coverage",
                           substitution_enabled=False, same_layer_alpha=.25, path_eta=.5)
    controller = GlobalExpertController(config, similarity, affinity)
    executor = LegacySlotExecutorAdapter(dispatcher, config.per_rank_slots()[rank], 0, 128)
    candidates = []
    captured = {}
    independent_totals = dict(local_token_rank_pairs=0, remote_token_rank_pairs=0,
                              local_expert_routes=0, remote_expert_routes=0)

    def observe(routes, plan, output):
        measured = accounting(routes.origin_ranks, plan.effective_token_routes, plan.owner_by_expert, world)
        for key in independent_totals:
            independent_totals[key] += measured[key]
            assert independent_totals[key] == getattr(runtime.metrics, key)
        if captured["decode"]:
            candidates.append(dict(layer=routes.layer, origins=routes.origin_ranks.copy(),
                selected=routes.selected_experts.copy(), weights=routes.routing_weights.copy(),
                probs=routes.full_router_probs.copy(), hidden=captured["hidden"].cpu(),
                expected=output.cpu(), active=len(set(routes.selected_experts.reshape(-1).tolist())),
                index=len(candidates)))

    runtime = DistributedMoERuntime(controller, executor, debug=True, observer=observe)
    original = runtime.forward_layer
    def capture(**kwargs):
        captured.update(decode=kwargs["is_decode"], hidden=kwargs["hidden_states"].detach())
        return original(**kwargs)
    runtime.forward_layer = capture
    attach_qwen3_runtime(model, runtime)
    with torch.inference_mode():
        benchmark.generate(model, encoded["input_ids"], encoded["attention_mask"], 9)
    detach_qwen3_runtime(model)
    capture_events = runtime.events
    ordered = sorted(candidates, key=lambda item: (item["active"], item["index"]))
    chosen = [ordered[0], ordered[(len(ordered) - 1) // 2], ordered[-1]]
    torch.save(chosen, root / f"events-rank{rank}.pt")
    selection = [{k: event[k] for k in ("layer", "active", "index")} for event in chosen]
    if rank == 0:
        (root / "selection.json").write_text(json.dumps({"pool": len(candidates), "events": selection}, indent=2) + "\n")
    del candidates, ordered
    results = []
    for quantile, event in zip(("low", "median", "high"), chosen):
        plans = [make_maps(event["origins"], event["selected"], world) if rank == 0 else None]
        dist.broadcast_object_list(plans, src=0, device=torch.device("cuda:0"))
        start = rank * 8
        hidden = event["hidden"].cuda()
        expected = event["expected"].cuda()
        routes = [{int(e): float(w) for e, w in zip(es, ws)}
                  for es, ws in zip(event["selected"], event["weights"])]
        for owner_map in plans[0]:
            dispatcher.reset_slot_pool(config.per_rank_slots()[rank])
            controller = GlobalExpertController(config, similarity, affinity)
            controller.admission = FixedAdmission(owner_map["owners"], owner_map["quotas"])
            executor = LegacySlotExecutorAdapter(dispatcher, config.per_rank_slots()[rank], 0, 128)
            runtime = DistributedMoERuntime(controller, executor, debug=True)
            with torch.inference_mode():
                loaded = runtime.forward_layer(event["layer"], hidden,
                    torch.as_tensor(event["selected"][start:start + 8], device="cuda"),
                    torch.as_tensor(event["weights"][start:start + 8], device="cuda", dtype=torch.bfloat16),
                    torch.as_tensor(event["probs"][start:start + 8], device="cuda"), True)
            torch.testing.assert_close(loaded, expected, atol=0, rtol=0)
            from mgo_v2.types import LayerRoutes
            logical = LayerRoutes(event["layer"], event["origins"], event["selected"], event["weights"], event["probs"])
            fixed_plan = controller.plan_layer(logical)
            assert fixed_plan.owner_by_expert == owner_map["owners"]
            assert all(not plan.miss_ops for plan in fixed_plan.local_exec.values())
            counts_before = dispatcher.get_cache_stats().tolist()

            def execute(stats=None):
                received = dispatch_tokens(hidden, routes[start:start + 8], owner_map["owners"], 8, stats)
                partials = executor.execute(event["layer"], received, fixed_plan.local_exec[rank], True)
                return return_partials(partials, received, 8, stats)

            # Primary layer wall latency excludes CUDA-event instrumentation.
            # Separate paired diagnostic passes collect communication intervals.
            records = []
            with torch.inference_mode():
                for _ in range(3):
                    actual = execute()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for instrumented in (False, True):
                    for iteration in range(args.iterations):
                        stats = CollectiveStats() if instrumented else None
                        dist.barrier()
                        torch.cuda.synchronize()
                        before = dispatcher.get_cache_stats().tolist()[3]
                        began = time.perf_counter()
                        actual = execute(stats)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - began
                        after = dispatcher.get_cache_stats().tolist()[3]
                        assert before == after, "expert H2D in resident timing"
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        receipt = dict(iteration=iteration, instrumented=instrumented,
                                       layer_seconds=elapsed, expert_fetches=after - before)
                        if stats:
                            receipt["collectives"] = stats.summary()
                        records.append(receipt)
            executor.assert_cache_matches(controller.cache.ranks[rank])
            assert dispatcher.get_cache_stats().tolist()[3] == counts_before[3]
            results.append(dict(event=quantile, selection={k: event[k] for k in ("layer", "active", "index")},
                **owner_map, records=records, bitwise_output_equal=True, timed_expert_h2d_bytes=0))
            print(json.dumps({"rank": rank, "event": quantile, "mode": owner_map["mode"],
                              "remote_fraction": owner_map["accounting"]["remote_pair_fraction"], "complete": True}), flush=True)
    sources = list((Path(__file__).parents[1] / "mgo_v2").glob("*.py")) + [Path(__file__), Path(__file__).with_name("locality_maps.py")]
    receipt = dict(status="PASS", rank=rank, world=world, boot=benchmark.BOOT, config=asdict(config),
        checkpoint=store["identity"], selections=selection, counter_oracle_events=capture_events,
        capture_counter_totals=independent_totals,
        input_hashes={name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
                      for name in ("workload", "similarity", "affinity")},
        source_hashes={str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sources}, results=results,
        scope="Resident dispatch/expert execution/combine; frozen plans exclude routing metadata/controller. Separate CUDA-event diagnostic passes; labels are swap heuristics, not proven global optima.")
    (root / f"rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    del runtime, executor, model, dispatcher
    handle.clean_up_resources()


if __name__ == "__main__":
    main()
