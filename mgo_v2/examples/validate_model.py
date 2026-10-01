#!/usr/bin/env python3
"""Full checkpoint EP parity and physical-cache audit against native receipts."""
import argparse
import faulthandler
import signal
import json
import os
from pathlib import Path

from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT = pin_rank_before_cuda_import()
os.environ.setdefault("MOE_INFINITY_MAX_TOKENS", "128")
os.environ.setdefault("MOE_IO_THREADS", "4")
os.environ.setdefault("MOE_INFINITY_DISABLE_MLOCK", "1")

import numpy as np
import torch
import torch.distributed as dist

from mgo_v2.communicator import warmup_collectives
from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.executor import LegacySlotExecutorAdapter
from mgo_v2.model_loader import load_qwen3_slots
from mgo_v2.qwen3_integration import attach_qwen3_runtime
from mgo_v2.runtime import DistributedMoERuntime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--offload-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-ratio", type=float, default=1.0)
    p.add_argument("--prompt-index", type=int, default=None)
    p.add_argument("--diagnostic", action="store_true", help="record divergences without a successful parity claim")
    args = p.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    warmup_collectives()
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"rank": rank, "boot": BOOT, "loading": True}), flush=True)
    model, handle, dispatcher, manifest = load_qwen3_slots(args.model, args.offload_dir)
    cfg = model.config
    config = RuntimeConfig(num_layers=cfg.num_hidden_layers, num_experts=cfg.num_experts,
                           top_k=cfg.num_experts_per_tok, world_size=world,
                           global_cache_ratio=args.cache_ratio, substitution_enabled=False,
                           eviction="lru", admission="random")
    controller = GlobalExpertController(config, np.tile(np.eye(cfg.num_experts, dtype=np.float32),
                                                       (cfg.num_hidden_layers, 1, 1)))
    executor = LegacySlotExecutorAdapter(dispatcher, config.per_rank_slots()[rank], 0, cfg.num_experts)
    runtime = DistributedMoERuntime(controller, executor, debug=True)
    patched = attach_qwen3_runtime(model, runtime)
    print(json.dumps({"rank": rank, "loaded": True, "patched": patched}), flush=True)
    reference_events = []
    event_metrics = []
    event_index = 0
    valid_only = False
    current_valid = None
    def hook(module, inputs, output):
        nonlocal event_index
        expected = reference_events[event_index]
        hidden, logits = output
        if valid_only:
            hidden = hidden.reshape(-1, hidden.shape[-1])[current_valid]
            logits = logits[current_valid]
        probs = logits.float().softmax(-1)
        weights, selected = probs.topk(module.top_k, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(hidden.dtype)
        target = expected["output"].to(hidden.device)
        error = (hidden.float() - target.float())
        row = {"event": event_index, "layer": module.layer_id,
               "selected_mismatches": int((selected.cpu() != expected["selected"]).sum()),
               "weight_max_error": float((weights.cpu().float() - expected["weights"].float()).abs().max()),
               "output_mismatches": int((hidden != target).sum()),
               "max_abs_error": error.abs().max().item(),
               "relative_l2": (error.norm() / target.float().norm().clamp_min(1e-12)).item(),
               "checksum": hidden.double().sum().item(), "native_checksum": expected["checksum"]}
        event_metrics.append(row)
        event_index += 1
    hooks = [layer.mlp.register_forward_hook(hook) for layer in model.model.layers]
    results = []
    prompt_indices = [args.prompt_index] if args.prompt_index is not None else (range(4) if world == 1 else [rank % 4])
    with torch.inference_mode():
        for prompt_index in prompt_indices:
            ref = torch.load(Path(args.reference) / f"prompt{prompt_index}.pt", weights_only=True)
            valid_only = ref.get("valid_only", False)
            reference_events = ref["events"]
            event_metrics = []
            event_index = 0
            all_ids = ref["input_ids"].cuda()
            mask = ref.get("attention_mask", torch.ones_like(ref["input_ids"])).cuda()
            past = None
            for step in range(len(ref["logits"])):
                current = all_ids if past is None else all_ids[:, -1:]
                positions = mask.long().cumsum(-1) - 1
                positions.masked_fill_(mask == 0, 0)
                current_valid = mask[:, -current.shape[1]:].reshape(-1).bool()
                out = model(input_ids=current, attention_mask=mask, position_ids=positions[:, -current.shape[1]:],
                            past_key_values=past, use_cache=True, logits_to_keep=1)
                all_ids = torch.cat((all_ids, out.logits[:, -1].argmax(-1, keepdim=True)), -1)
                past = out.past_key_values
                mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), 1)
            result = {"prompt_index": prompt_index, "token_ids": all_ids.cpu().tolist(),
                      "tokens_equal": bool(torch.equal(all_ids.cpu(), ref["output_ids"])),
                      "selected_mismatches": sum(row["selected_mismatches"] for row in event_metrics),
                      "output_mismatches": sum(row["output_mismatches"] for row in event_metrics),
                      "weight_max_error": max(row["weight_max_error"] for row in event_metrics),
                      "max_relative_l2": max(row["relative_l2"] for row in event_metrics),
                      "events": list(event_metrics)}
            result["status"] = "PASS" if (result["tokens_equal"] and result["selected_mismatches"] == 0
                                              and result["weight_max_error"] == 0 and result["output_mismatches"] == 0) else "FAIL"
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "events"}), flush=True)
            # Persist each completed prompt even if a later prompt fails.
            (root / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "world": world,
                "boot": BOOT, "cache_ratio": args.cache_ratio, "prompts": results,
                "metrics": runtime.metrics.to_dict(), "controller_seconds": runtime.controller_seconds,
                "cache_stats": dispatcher.get_cache_stats().tolist(),
                "phase_times_us": dispatcher.get_phase_times().tolist()}, indent=2) + "\n")
    for item in hooks:
        item.remove()
    dist.barrier()
    failed = torch.tensor([int(any(row["status"] != "PASS" for row in results))], device="cuda")
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    any_failed = bool(failed.item())
    dist.destroy_process_group()
    # Modules retain the runtime and dispatcher, so release model first.
    del model, runtime, executor, dispatcher
    handle.clean_up_resources()
    if any_failed and not args.diagnostic:
        raise AssertionError("native exact parity failed; inspect the saved per-layer receipts")


if __name__ == "__main__":
    main()
