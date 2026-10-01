"""Fair-comparison variant of wikitext_qwen3_phase_metrics.py.

Differences vs the original:

1. `max_input_tokens` is held CONSTANT across all batch sizes (no dynamic
   shrinking based on `moe_max_prefill_tokens // batch_size`). The CLI default
   picks a value safe for the largest batch under the current kMaxTokens.
2. Prompts are padded to `padding="max_length"` so `prompt_tokens_padded` is
   identical for every batch size (every sequence is exactly
   `max_input_tokens` tokens, regardless of source text length).
3. A warmup pass runs before the measurement loop. The warmup uses bs=1
   because the C++ dispatcher's batch_size==1 branch in
   ExpertDispatcher::GPUExecFunc avoids the multi-GPU index() path that
   crashes when triggered on a cold engine. After bs=1 has primed the engine,
   subsequent measurements (including bs>1) reuse warm dispatcher state.
4. Dispatcher stats are reset before EACH measured batch so per-row counters
   are isolated (the original only reset once at the beginning).
5. Archer metric columns are still recorded for transparency but are
   structurally zero for Qwen3 (the dispatch_local path bypasses
   AcquireTensor, so topology-node visit counters never increment).
"""

import argparse
import csv
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from moe_infinity import MoE


METRIC_COLUMNS = {
    "visit": 0,
    "gpu_visit": 1,
    "cpu_visit": 2,
    "hit": 3,
    "gpu_hit": 4,
    "cpu_hit": 5,
    "tensor_count": 6,
    "prefetch": 7,
    "unused": 8,
    "io_state": 9,
    "is_sparse": 10,
}

DISPATCHER_COLUMNS = {
    "visit": 0,
    "hit": 1,
    "miss": 2,
    "gpu_fetch": 3,
    "eviction": 4,
    "overload_fetch": 5,
}


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def pick_nonempty_texts(dataset, count: int, max_chars: int) -> list[str]:
    texts = []
    for row in dataset:
        text = row["text"].strip()
        if text:
            texts.append(text[:max_chars])
        if len(texts) >= count:
            break
    if len(texts) < count:
        raise RuntimeError(f"Only found {len(texts)} non-empty WikiText rows.")
    return texts


def make_prompt(text: str) -> str:
    return (
        "Continue the following WikiText passage in the same style.\n\n"
        f"{text}\n\nContinuation:"
    )


def build_batch(tokenizer, prompts: list[str], max_input_tokens: int):
    if getattr(tokenizer, "chat_template", None):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
    else:
        texts = prompts

    return tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_input_tokens,
    )


def snapshot_archer_metrics(moe_model) -> torch.Tensor:
    return moe_model.engine.archer_engine.get_hit_rate().cpu().to(torch.int64)


def snapshot_dispatcher_metrics(moe_model) -> torch.Tensor:
    dispatcher = getattr(moe_model.engine, "expert_dispatcher", None)
    if dispatcher is None or not hasattr(dispatcher, "get_cache_stats"):
        return torch.zeros(len(DISPATCHER_COLUMNS), dtype=torch.int64)
    return dispatcher.get_cache_stats().cpu().to(torch.int64)


def summarize_archer_delta(before: torch.Tensor, after: torch.Tensor, prefix: str) -> dict:
    delta = after - before
    sparse_mask = after[:, METRIC_COLUMNS["is_sparse"]] == 1
    sparse_delta = delta[sparse_mask]

    def col_sum(name: str) -> int:
        if sparse_delta.numel() == 0:
            return 0
        return int(sparse_delta[:, METRIC_COLUMNS[name]].sum().item())

    gpu_visit = col_sum("gpu_visit")
    hit = col_sum("hit")
    miss = max(gpu_visit - hit, 0)
    prefetch_count = col_sum("prefetch")

    return {
        f"{prefix}_archer_sparse_visit": col_sum("visit"),
        f"{prefix}_archer_sparse_gpu_visit": gpu_visit,
        f"{prefix}_archer_sparse_cpu_visit": col_sum("cpu_visit"),
        f"{prefix}_archer_cache_hit": hit,
        f"{prefix}_archer_cache_miss": miss,
        f"{prefix}_archer_cache_hit_ratio": hit / gpu_visit if gpu_visit else None,
        f"{prefix}_archer_cache_miss_ratio": miss / gpu_visit if gpu_visit else None,
        f"{prefix}_prefetch_count": prefetch_count,
        f"{prefix}_prefetch_hit_ratio": None,
        f"{prefix}_prefetch_miss_ratio": None,
    }


def summarize_dispatcher_delta(before: torch.Tensor, after: torch.Tensor, prefix: str) -> dict:
    delta = after - before

    def value(name: str) -> int:
        return int(delta[DISPATCHER_COLUMNS[name]].item())

    visit = value("visit")
    hit = value("hit")
    miss = value("miss")

    return {
        f"{prefix}_cache_visit": visit,
        f"{prefix}_cache_hit": hit,
        f"{prefix}_cache_miss": miss,
        f"{prefix}_cache_hit_ratio": hit / visit if visit else None,
        f"{prefix}_cache_miss_ratio": miss / visit if visit else None,
        f"{prefix}_gpu_fetch": value("gpu_fetch"),
        f"{prefix}_evictions": value("eviction"),
        f"{prefix}_overload_fetches": value("overload_fetch"),
    }


def phase_snapshot(moe_model) -> tuple[torch.Tensor, torch.Tensor]:
    return snapshot_archer_metrics(moe_model), snapshot_dispatcher_metrics(moe_model)


def summarize_phase(
    moe_model,
    before: tuple[torch.Tensor, torch.Tensor],
    after: tuple[torch.Tensor, torch.Tensor],
    prefix: str,
) -> dict:
    archer_before, dispatcher_before = before
    archer_after, dispatcher_after = after
    row = {}
    row.update(summarize_archer_delta(archer_before, archer_after, prefix))
    row.update(summarize_dispatcher_delta(dispatcher_before, dispatcher_after, prefix))
    return row


def write_outputs(output_dir: str, rows: list[dict]):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "phase_batch_metrics.json")
    csv_path = os.path.join(output_dir, "phase_batch_metrics.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"metrics_json={json_path}", flush=True)
    print(f"metrics_csv={csv_path}", flush=True)


def run_one_batch(
    moe_model,
    hf_model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> dict:
    batch_size = int(input_ids.shape[0])
    prompt_tokens = int(input_ids.shape[1])

    moe_model._configure_hook(input_ids)
    hf_model.eval()

    torch.cuda.synchronize()
    prefill_before = phase_snapshot(moe_model)
    prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_start
    prefill_after = phase_snapshot(moe_model)

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    generated_tokens = [next_token.detach().cpu()]

    decode_times = []
    decode_before = phase_snapshot(moe_model)
    for _ in range(max(max_new_tokens - 1, 0)):
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        torch.cuda.synchronize()
        token_start = time.perf_counter()
        with torch.no_grad():
            outputs = hf_model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - token_start)

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        generated_tokens.append(next_token.detach().cpu())
    decode_after = phase_snapshot(moe_model)

    decode_seconds = sum(decode_times)
    decode_tokens = max(max_new_tokens - 1, 0)
    e2e_seconds = prefill_seconds + decode_seconds
    generated = torch.cat(generated_tokens, dim=1)

    row = {
        "batch_size": batch_size,
        "prompt_tokens_padded": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "decode_tokens_measured": decode_tokens,
        "prefill_seconds": prefill_seconds,
        "ttft_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "tpot_seconds": decode_seconds / decode_tokens if decode_tokens else None,
        "e2e_latency_seconds": e2e_seconds,
        "tokens_per_second_e2e": (batch_size * max_new_tokens) / e2e_seconds
        if e2e_seconds > 0
        else None,
        "generated_token_ids_head": generated[: min(batch_size, 4)].tolist(),
    }
    row.update(summarize_phase(moe_model, prefill_before, prefill_after, "prefill"))
    row.update(summarize_phase(moe_model, decode_before, decode_after, "decode"))
    return row


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fair-comparison Qwen3 MoE-Infinity prefill/decode benchmark. "
            "Holds max_input_tokens constant across all batch sizes and uses a "
            "warmup pass so every measurement starts from a warm cache."
        )
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
    parser.add_argument(
        "--output_dir",
        default="/tmp/moe_infinity_metrics/qwen3_wikitext_phase_bs1_64_cache50_fair",
    )
    parser.add_argument("--batch_sizes", type=parse_batch_sizes, default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_chars", type=int, default=64)
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=16,
        help=(
            "Held constant for every batch size. Default 16 keeps "
            "max_batch * max_input_tokens <= kMaxTokens=1024 even at bs=64."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--device_memory_ratio", type=float, default=0.50)
    parser.add_argument(
        "--warmup_batch_size",
        type=int,
        default=1,
        help=(
            "Batch size used for the warmup pass. Default 1, which avoids the "
            "multi-GPU index() path in ExpertDispatcher::GPUExecFunc that "
            "crashes on a cold engine."
        ),
    )
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip the warmup pass (useful for debugging).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process.")

    warmup_bs = args.warmup_batch_size
    print(f"batch_sizes={args.batch_sizes}", flush=True)
    print(f"max_new_tokens={args.max_new_tokens}", flush=True)
    print(f"max_input_tokens={args.max_input_tokens} (fixed for all batches)", flush=True)
    print(f"warmup_batch_size={warmup_bs} (skip={args.skip_warmup})", flush=True)
    print(f"device_memory_ratio={args.device_memory_ratio}", flush=True)
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"torch_cuda_devices={torch.cuda.device_count()}", flush=True)

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=args.split,
        cache_dir=args.dataset_cache,
    )

    # Pull enough texts to cover the largest measurement batch AND the warmup
    # batch, choosing the larger.
    texts_needed = max(max(args.batch_sizes), warmup_bs)
    texts = pick_nonempty_texts(dataset, texts_needed, args.max_chars)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = os.path.basename(os.path.normpath(args.model_path))
    config = {
        "offload_path": os.path.join(args.offload_dir, model_name),
        "device_memory_ratio": args.device_memory_ratio,
    }

    load_start = time.perf_counter()
    moe_model = MoE(args.model_path, config)
    load_seconds = time.perf_counter() - load_start
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    dispatcher = getattr(moe_model.engine, "expert_dispatcher", None)

    # Warmup pass: populate the dispatcher cache so every measured batch runs
    # against the same warm-cache steady state. Without this, the first
    # measured batch is unfairly penalized by cold-cache misses.
    if not args.skip_warmup:
        warmup_prompts = [make_prompt(text) for text in texts[:warmup_bs]]
        warmup_encoded = build_batch(tokenizer, warmup_prompts, args.max_input_tokens)
        warmup_ids = warmup_encoded["input_ids"].to("cuda:0")
        warmup_mask = warmup_encoded["attention_mask"].to("cuda:0")
        print(
            f"warmup_pass batch_size={warmup_bs} "
            f"input_tokens={warmup_ids.shape[1]}",
            flush=True,
        )
        warmup_start = time.perf_counter()
        _ = run_one_batch(
            moe_model=moe_model,
            hf_model=moe_model.model,
            tokenizer=tokenizer,
            input_ids=warmup_ids,
            attention_mask=warmup_mask,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            f"warmup_done seconds={time.perf_counter() - warmup_start:.3f}",
            flush=True,
        )

    rows = []
    for batch_size in args.batch_sizes:
        prompts = [make_prompt(text) for text in texts[:batch_size]]
        print(
            f"running_batch_size={batch_size} "
            f"max_input_tokens={args.max_input_tokens}",
            flush=True,
        )
        encoded = build_batch(tokenizer, prompts, args.max_input_tokens)
        input_ids = encoded["input_ids"].to("cuda:0")
        attention_mask = encoded["attention_mask"].to("cuda:0")

        # Reset dispatcher stats so per-row counters are isolated. Cache
        # contents stay warm (intentional, see module docstring).
        if dispatcher is not None and hasattr(dispatcher, "reset_cache_stats"):
            dispatcher.reset_cache_stats()

        row = run_one_batch(
            moe_model=moe_model,
            hf_model=moe_model.model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
        )
        row["model_load_seconds"] = load_seconds
        row["max_input_tokens_effective"] = args.max_input_tokens
        row["warmup_batch_size"] = warmup_bs if not args.skip_warmup else None
        rows.append(row)
        print("metric_row=" + json.dumps(row, default=str), flush=True)
        write_outputs(args.output_dir, rows)

    write_outputs(args.output_dir, rows)


if __name__ == "__main__":
    main()
