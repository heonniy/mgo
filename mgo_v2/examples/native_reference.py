#!/usr/bin/env python3
"""Save fixed-prompt native Qwen3 routes, per-layer tensors and greedy tokens."""
import argparse
import json
from pathlib import Path

from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT = pin_rank_before_cuda_import()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["What is 17 + 25?", "The capital of France is", "Explain why the sky appears blue.",
           "Write a Python function that adds two numbers."]


def capture_hook(layer, events):
    def hook(module, inputs, output):
        hidden, logits = output
        probs = torch.softmax(logits.float(), -1)
        weights, ids = probs.topk(module.top_k, dim=-1)
        if module.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        events.append({"layer": layer, "input": inputs[0].detach().cpu(),
                       "selected": ids.cpu(), "weights": weights.to(hidden.dtype).cpu(),
                       "output": hidden.detach().cpu(),
                       "checksum": hidden.double().sum().item()})
    return hook


def generate_steps(model, input_ids, steps):
    all_ids = input_ids
    past = None
    logits = []
    for step in range(steps):
        out = model(input_ids=all_ids if past is None else all_ids[:, -1:],
                    past_key_values=past, use_cache=True)
        logits.append(out.logits[:, -1].float().cpu())
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        all_ids = torch.cat((all_ids, token), dim=1)
        past = out.past_key_values
    return all_ids, logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=4)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="eager", local_files_only=True).eval()
    events = []
    hooks = [layer.mlp.register_forward_hook(capture_hook(i, events))
             for i, layer in enumerate(model.model.layers)]
    summary = {"model": str(Path(args.model).resolve()), "boot": BOOT, "prompts": []}
    with torch.inference_mode():
        for i, prompt in enumerate(PROMPTS):
            events.clear()
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            output, logits = generate_steps(model, ids, args.steps)
            torch.save({"input_ids": ids.cpu(), "output_ids": output.cpu(), "events": list(events),
                        "logits": logits}, root / f"prompt{i}.pt")
            row = {"prompt": prompt, "token_ids": output[0].tolist(), "text": tok.decode(output[0]),
                   "events": len(events)}
            summary["prompts"].append(row)
            print(json.dumps(row), flush=True)
    for hook in hooks:
        hook.remove()
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
