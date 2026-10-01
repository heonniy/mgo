#!/usr/bin/env python3
"""Checkpoint-backed, resumable model measurements with explicit cell manifests.

Each cell starts with empty expert residency and a fresh controller. Generated
work is a fixed number of steps on all ranks; EOS is respected when scoring,
but remaining steps still execute so collective order cannot diverge.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import re
import time
from decimal import Decimal
from pathlib import Path

from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT = pin_rank_before_cuda_import()

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer
from mgo_v2.affinity import AffinityTables
from mgo_v2.communicator import warmup_collectives
from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.executor import LegacySlotExecutorAdapter
from mgo_v2.model_loader import load_qwen3_slots
from mgo_v2.profiling import CollectiveStats
from mgo_v2.qwen3_integration import attach_qwen3_runtime, detach_qwen3_runtime
from mgo_v2.runtime import DistributedMoERuntime


def numeric_answer(text):
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return format(Decimal(matches[-1].replace(",", "")).normalize(), "f") if matches else None


def generate(model, ids, mask, steps, after_prefill=None):
    """Time complete forward + greedy selection, synchronized at each token."""
    past, output, times = None, [], []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 0)
        out = model(input_ids=ids, attention_mask=mask, position_ids=positions[:, -ids.shape[1]:],
                    past_key_values=past, use_cache=True, logits_to_keep=1)
        ids = out.logits[:, -1].argmax(-1, keepdim=True)
        past = out.past_key_values
        output.append(ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if after_prefill is not None and len(output) == 1:
            after_prefill()
        mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), 1)
    return torch.cat(output, 1), times


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "offload-dir", "similarity", "affinity", "workload", "cells", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--debug-cache", action="store_true")
    p.add_argument("--profile", action="store_true", help="instrument collective CUDA intervals; affects timings")
    args = p.parse_args()
    if args.steps < 2 or args.repeats < 1:
        p.error("steps must be >=2 and repeats >=1")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    warmup_collectives()
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cells = json.loads(Path(args.cells).read_text())
    workload = json.loads(Path(args.workload).read_text())
    similarity = np.load(args.similarity)
    affinity = AffinityTables.load_npz(args.affinity)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    model, handle, dispatcher, store_manifest = load_qwen3_slots(args.model, args.offload_dir)
    runtime = executor = None
    cfg = model.config
    code_hash = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    for path in sorted((package_root / "mgo_v2").glob("*.py")) + [Path(__file__)]:
        code_hash.update(path.name.encode())
        code_hash.update(path.read_bytes())
    extension_path, = (package_root.parent / "MoE-Infinity-EP-archer-coslot" / "moe_infinity").glob("_store*.so")
    code_hash.update(extension_path.read_bytes())
    provenance = {"boot": BOOT, "world": world, "rank": rank, "steps": args.steps,
                  "repeats": args.repeats, "instrumented": args.profile,
                  "checkpoint": store_manifest["identity"],
                  "code_sha256": code_hash.hexdigest(),
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
                  "input_sha256": {name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
                                   for name in ("similarity", "affinity", "workload", "cells")},
                  "quality_scope": "Short zero-shot numeric GSM8K screen, not the prior few-shot quality protocol",
                  "timing_scope": "Empty expert cache; fixed-step generation; includes controller and communication, excludes loading/tokenization"}
    provenance_path = root / f"provenance-rank{rank}.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("existing results have different code, inputs or settings; use a fresh output directory")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    # Warm CUDA kernels once. Residency/history are reset before every cell.
    warmed = False
    for cell in cells:
        label = cell["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", label):
            raise ValueError("unsafe cell name")
        batch = cell["batch"]
        if batch * world > len(workload):
            raise ValueError("workload must provide a distinct question for every global batch row")
        rows = workload[rank * batch:(rank + 1) * batch]
        texts = [tok.apply_chat_template([
            {"role": "user", "content": row["question"] + "\nReturn only the final numeric answer, without explanation."}],
            tokenize=False, add_generation_prompt=True) for row in rows]
        encoded = tok(texts, padding=True, return_tensors="pt")
        encoded = {k: v.cuda() for k, v in encoded.items()}
        for repeat in range(args.repeats):
            path = root / f"{label}-rep{repeat}-rank{rank}.json"
            done = torch.tensor([int(path.exists())], device="cuda")
            dist.all_reduce(done, op=dist.ReduceOp.MIN)
            if done.item():
                continue
            config = RuntimeConfig(num_layers=cfg.num_hidden_layers, num_experts=cfg.num_experts,
                top_k=cfg.num_experts_per_tok, world_size=world, **cell["config"])
            detach_qwen3_runtime(model)
            dispatcher.reset_slot_pool(config.per_rank_slots()[rank])
            controller = GlobalExpertController(config, similarity, affinity)
            executor = LegacySlotExecutorAdapter(dispatcher, config.per_rank_slots()[rank], 0, cfg.num_experts)
            collective_stats = CollectiveStats() if args.profile else None
            runtime = DistributedMoERuntime(controller, executor, debug=args.debug_cache,
                                            collective_stats=collective_stats)
            attach_qwen3_runtime(model, runtime)
            if not warmed:
                with torch.inference_mode():
                    generate(model, encoded["input_ids"][:, -8:], encoded["attention_mask"][:, -8:], 2)
                warmed = True
                # Restart controller and physical slots together; model weights stay loaded.
                detach_qwen3_runtime(model)
                dispatcher.reset_slot_pool(config.per_rank_slots()[rank])
                controller = GlobalExpertController(config, similarity, affinity)
                executor = LegacySlotExecutorAdapter(dispatcher, config.per_rank_slots()[rank], 0, cfg.num_experts)
                collective_stats = CollectiveStats() if args.profile else None
                runtime = DistributedMoERuntime(controller, executor, debug=args.debug_cache,
                                                collective_stats=collective_stats)
                attach_qwen3_runtime(model, runtime)
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()
            print(json.dumps({"cell": label, "rank": rank, "repeat": repeat, "started": True}), flush=True)
            prefill = {}
            def snapshot_prefill():
                prefill.update(metrics=runtime.metrics.to_dict(), controller_seconds=runtime.controller_seconds,
                               cache_stats=dispatcher.get_cache_stats().tolist())
                if collective_stats:
                    prefill["peer_payload_tx_bytes"] = dict(collective_stats.payload_bytes)
            dist.barrier()
            torch.cuda.synchronize()
            generation_start = time.perf_counter()
            with torch.inference_mode():
                output, seconds = generate(model, encoded["input_ids"], encoded["attention_mask"], args.steps, snapshot_prefill)
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_start
            executor.assert_cache_matches(controller.cache.ranks[rank])
            samples = []
            for row, ids in zip(rows, output.cpu().tolist()):
                finished = eos in ids
                usable = ids[:ids.index(eos)] if finished else ids
                text = tok.decode(usable, skip_special_tokens=True)
                prediction = numeric_answer(text)
                reference = numeric_answer(row["reference"])
                samples.append({"sample_id": row["sample_id"], "text": text, "token_ids": usable,
                                "finished": finished, "prediction": prediction, "reference": reference,
                                "correct": finished and prediction is not None and prediction == reference})
            timings = torch.tensor(seconds, device="cuda", dtype=torch.float64)
            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
            max_seconds = timings.cpu().tolist()
            elapsed = torch.tensor(generation_seconds, device="cuda", dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            global_generation_seconds = elapsed.item()
            result = {"status": "PASS", "name": label, "rank": rank, "world": world, "repeat": repeat,
                      "local_batch": batch, "prompt_tokens": encoded["attention_mask"].sum(-1).cpu().tolist(),
                      "padded_prompt_tokens": encoded["input_ids"].shape[1], "config": asdict(config),
                      "rank_step_seconds": seconds, "global_max_step_seconds": max_seconds,
                      "ttft_seconds": max_seconds[0], "tpot_seconds": sum(max_seconds[1:]) / (args.steps - 1),
                      "rank_generation_seconds": generation_seconds,
                      "global_max_generation_seconds": global_generation_seconds,
                      "fixed_step_output_tokens_per_second": batch * world * args.steps / global_generation_seconds,
                      "metrics": runtime.metrics.to_dict(), "controller_seconds": runtime.controller_seconds,
                      "prefill": prefill,
                      "cache_stats": dispatcher.get_cache_stats().tolist(),
                      "fetch_modes": dispatcher.get_fetch_mode_counts().tolist(),
                      "generated_token_ids": output.cpu().tolist(),
                      "host_fetch_bytes": dispatcher.get_cache_stats()[3].item() * cfg.hidden_size * cfg.moe_intermediate_size * 3 * 2,
                      "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "quality": {"correct": sum(row["correct"] for row in samples), "total": len(samples),
                                  "unfinished": sum(not row["finished"] for row in samples), "samples": samples}}
            if collective_stats:
                result["collectives"] = collective_stats.summary()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(path)
            print(json.dumps({"cell": label, "rank": rank, "repeat": repeat, "complete": True,
                              "ttft_seconds": result["ttft_seconds"], "tpot_seconds": result["tpot_seconds"]}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    detach_qwen3_runtime(model)
    del model, runtime, executor, dispatcher
    handle.clean_up_resources()


if __name__ == "__main__":
    main()
