#!/usr/bin/env python3
"""Native HF control for the short numeric quality screen in benchmark_model."""
import argparse
import hashlib
import json
from pathlib import Path

# Imports the same generation/scoring helpers and pins visibility before CUDA.
from benchmark_model import BOOT, generate, numeric_answer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "workload", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("use a fresh reference output directory")
    rows = json.loads(Path(args.workload).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="eager", local_files_only=True).eval()
    samples = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            batch = rows[start:start + args.batch]
            texts = [tokenizer.apply_chat_template([
                {"role": "user", "content": row["question"] + "\nReturn only the final numeric answer, without explanation."}],
                tokenize=False, add_generation_prompt=True) for row in batch]
            encoded = {k:v.cuda() for k,v in tokenizer(texts, padding=True, return_tensors="pt").items()}
            output, _ = generate(model, encoded["input_ids"], encoded["attention_mask"], args.steps)
            for row, ids in zip(batch, output.cpu().tolist()):
                finished = tokenizer.eos_token_id in ids
                usable = ids[:ids.index(tokenizer.eos_token_id)] if finished else ids
                text = tokenizer.decode(usable, skip_special_tokens=True)
                prediction, reference = numeric_answer(text), numeric_answer(row["reference"])
                samples.append({"sample_id": row["sample_id"], "text": text, "token_ids": usable,
                                "prediction": prediction, "reference": reference, "finished": finished,
                                "correct": finished and prediction is not None and prediction == reference})
            (root / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
            print(json.dumps({"completed": len(samples), "correct": sum(x["correct"] for x in samples)}), flush=True)
    result = {"status": "PASS", "model": args.model, "boot": BOOT, "steps": args.steps, "batch": args.batch,
              "workload_sha256": hashlib.sha256(Path(args.workload).read_bytes()).hexdigest(),
              "correct": sum(x["correct"] for x in samples), "total": len(samples),
              "unfinished": sum(not x["finished"] for x in samples),
              "scope": "Native BF16, eager GPU attention, short zero-shot numeric screen on GSM8K-train questions."}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
