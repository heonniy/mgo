import argparse
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from moe_infinity import MoE


def pick_nonempty_text(dataset, start_index, max_chars):
    for row in dataset.select(range(start_index, len(dataset))):
        text = row["text"].strip()
        if text:
            return text[:max_chars]
    raise RuntimeError("No non-empty WikiText row was found.")


def main():
    parser = argparse.ArgumentParser(
        description="Run a tiny WikiText sample through MoE-Infinity."
    )
    parser.add_argument(
        "--model_path",
        default="/home/work/hyewon.lee/model/Qwen3-235B-A22B",
    )
    parser.add_argument(
        "--dataset_cache",
        default="/home/work/hyewon.lee/dataset",
    )
    parser.add_argument(
        "--offload_dir",
        default="/tmp/moe_infinity_offload",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_chars", type=int, default=700)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--device_memory_ratio", type=float, default=0.75)
    args = parser.parse_args()

    os.makedirs(args.offload_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(args.model_path))
    offload_path = os.path.join(args.offload_dir, model_name)

    print(f"model_path={args.model_path}", flush=True)
    print(f"dataset=wikitext/wikitext-2-raw-v1", flush=True)
    print(f"dataset_cache={args.dataset_cache}", flush=True)
    print(f"offload_path={offload_path}", flush=True)
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"torch_cuda_devices={torch.cuda.device_count()}", flush=True)

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=args.split,
        cache_dir=args.dataset_cache,
    )
    text = pick_nonempty_text(dataset, args.sample_index, args.max_chars)
    prompt = (
        "Continue the following WikiText passage in the same style.\n\n"
        f"{text}\n\nContinuation:"
    )
    print(f"split={args.split}", flush=True)
    print(f"input_text_chars={len(text)}", flush=True)
    print("prompt_begin", flush=True)
    print(prompt, flush=True)
    print("prompt_end", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = {
        "offload_path": offload_path,
        "device_memory_ratio": args.device_memory_ratio,
    }
    load_start = time.time()
    model = MoE(args.model_path, config)
    print(f"model_load_seconds={time.time() - load_start:.3f}", flush=True)

    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        input_text = prompt

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")
    print(f"input_tokens={input_ids.shape[-1]}", flush=True)

    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(f"generate_seconds={time.time() - gen_start:.3f}", flush=True)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("decoded_begin", flush=True)
    print(decoded, flush=True)
    print("decoded_end", flush=True)


if __name__ == "__main__":
    main()
