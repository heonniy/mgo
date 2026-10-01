import argparse
import csv
import json
import os
import threading
import time
from dataclasses import dataclass, field

import pynvml
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

PCIE_MB_PER_SEC_PER_LANE = {
    1: 250.0,
    2: 500.0,
    3: 984.6,
    4: 1969.0,
    5: 3938.0,
    6: 7877.0,
}


@dataclass
class PcieStats:
    samples: int = 0
    avg_rx_kbps: float = 0.0
    avg_tx_kbps: float = 0.0
    max_rx_kbps: int = 0
    max_tx_kbps: int = 0
    approx_rx_mb: float = 0.0
    approx_tx_mb: float = 0.0
    rx_util_pct_avg: float | None = None
    tx_util_pct_avg: float | None = None
    rx_util_pct_max: float | None = None
    tx_util_pct_max: float | None = None


class PcieSampler:
    def __init__(self, gpu_index: int = 0, interval: float = 0.2):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[tuple[float, int, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.link_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(self.handle)
        self.link_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(self.handle)

    def theoretical_kbps(self) -> float | None:
        mbps_lane = PCIE_MB_PER_SEC_PER_LANE.get(self.link_gen)
        if mbps_lane is None:
            return None
        return mbps_lane * self.link_width * 1024.0

    def _run(self):
        while not self._stop.is_set():
            ts = time.time()
            rx = pynvml.nvmlDeviceGetPcieThroughput(
                self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
            )
            tx = pynvml.nvmlDeviceGetPcieThroughput(
                self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
            )
            self.samples.append((ts, rx, tx))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> PcieStats:
        if not self.samples:
            return PcieStats()

        rx_values = [sample[1] for sample in self.samples]
        tx_values = [sample[2] for sample in self.samples]
        avg_rx = sum(rx_values) / len(rx_values)
        avg_tx = sum(tx_values) / len(tx_values)

        approx_rx_mb = 0.0
        approx_tx_mb = 0.0
        for (t0, rx, tx), (t1, _, _) in zip(self.samples, self.samples[1:]):
            dt = max(t1 - t0, 0.0)
            approx_rx_mb += rx * dt / 1024.0
            approx_tx_mb += tx * dt / 1024.0

        denom = self.theoretical_kbps()
        rx_util_avg = tx_util_avg = rx_util_max = tx_util_max = None
        if denom and denom > 0:
            rx_util_avg = avg_rx / denom * 100.0
            tx_util_avg = avg_tx / denom * 100.0
            rx_util_max = max(rx_values) / denom * 100.0
            tx_util_max = max(tx_values) / denom * 100.0

        return PcieStats(
            samples=len(self.samples),
            avg_rx_kbps=avg_rx,
            avg_tx_kbps=avg_tx,
            max_rx_kbps=max(rx_values),
            max_tx_kbps=max(tx_values),
            approx_rx_mb=approx_rx_mb,
            approx_tx_mb=approx_tx_mb,
            rx_util_pct_avg=rx_util_avg,
            tx_util_pct_avg=tx_util_avg,
            rx_util_pct_max=rx_util_max,
            tx_util_pct_max=tx_util_max,
        )


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


def snapshot_metrics(model) -> torch.Tensor:
    return model.engine.archer_engine.get_hit_rate().cpu().to(torch.int64)


def snapshot_dispatcher_metrics(model) -> torch.Tensor:
    dispatcher = getattr(model.engine, "expert_dispatcher", None)
    if dispatcher is None or not hasattr(dispatcher, "get_cache_stats"):
        return torch.zeros(len(DISPATCHER_COLUMNS), dtype=torch.int64)
    return dispatcher.get_cache_stats().cpu().to(torch.int64)


def summarize_metric_delta(before: torch.Tensor, after: torch.Tensor) -> dict:
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
    prefetch = col_sum("prefetch")
    visit = col_sum("visit")
    cpu_visit = col_sum("cpu_visit")

    return {
        "sparse_visit": visit,
        "sparse_gpu_visit": gpu_visit,
        "sparse_cpu_visit": cpu_visit,
        "expert_cache_hit": hit,
        "expert_cache_miss": miss,
        "expert_cache_hit_ratio": hit / gpu_visit if gpu_visit else None,
        "expert_cache_miss_ratio": miss / gpu_visit if gpu_visit else None,
        "prefetch_count": prefetch,
        "sparse_nodes_touched": int((sparse_delta[:, METRIC_COLUMNS["visit"]] > 0).sum().item())
        if sparse_delta.numel()
        else 0,
    }


def summarize_dispatcher_delta(before: torch.Tensor, after: torch.Tensor) -> dict:
    delta = after - before

    def value(name: str) -> int:
        return int(delta[DISPATCHER_COLUMNS[name]].item())

    visit = value("visit")
    hit = value("hit")
    miss = value("miss")

    return {
        "dispatcher_expert_visit": visit,
        "dispatcher_cache_hit": hit,
        "dispatcher_cache_miss": miss,
        "dispatcher_cache_hit_ratio": hit / visit if visit else None,
        "dispatcher_cache_miss_ratio": miss / visit if visit else None,
        "dispatcher_gpu_fetch": value("gpu_fetch"),
        "dispatcher_evictions": value("eviction"),
        "dispatcher_overload_fetches": value("overload_fetch"),
    }


def build_batch_texts(tokenizer, prompts: list[str]) -> dict[str, torch.Tensor]:
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
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)


def write_outputs(output_dir: str, rows: list[dict]):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "batch_metrics.json")
    csv_path = os.path.join(output_dir, "batch_metrics.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"metrics_json={json_path}", flush=True)
    print(f"metrics_csv={csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Measure MoE-Infinity expert cache and PCIe metrics by batch size."
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
        default="/tmp/moe_infinity_metrics/qwen3_wikitext",
    )
    parser.add_argument("--batch_sizes", type=parse_batch_sizes, default=[1, 2])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_chars", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--device_memory_ratio", type=float, default=0.55)
    parser.add_argument("--pcie_sample_interval", type=float, default=0.2)
    args = parser.parse_args()

    print(f"batch_sizes={args.batch_sizes}", flush=True)
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"torch_cuda_devices={torch.cuda.device_count()}", flush=True)

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=args.split,
        cache_dir=args.dataset_cache,
    )
    texts = pick_nonempty_texts(dataset, max(args.batch_sizes), args.max_chars)

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

    load_start = time.time()
    model = MoE(args.model_path, config)
    load_seconds = time.time() - load_start
    print(f"model_load_seconds={load_seconds:.3f}", flush=True)

    rows = []
    for batch_size in args.batch_sizes:
        prompts = [make_prompt(text) for text in texts[:batch_size]]
        encoded = build_batch_texts(tokenizer, prompts)
        input_ids = encoded["input_ids"].to("cuda:0")
        attention_mask = encoded["attention_mask"].to("cuda:0")

        torch.cuda.synchronize()
        before = snapshot_metrics(model)
        dispatcher_before = snapshot_dispatcher_metrics(model)
        start = time.time()
        with PcieSampler(0, args.pcie_sample_interval) as sampler:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
        elapsed = time.time() - start
        after = snapshot_metrics(model)
        dispatcher_after = snapshot_dispatcher_metrics(model)
        metric_summary = summarize_metric_delta(before, after)
        dispatcher_summary = summarize_dispatcher_delta(
            dispatcher_before, dispatcher_after
        )
        pcie = sampler.summary()
        mem = pynvml.nvmlDeviceGetMemoryInfo(sampler.handle)

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_tail = [text[-80:].replace("\n", "\\n") for text in decoded]

        row = {
            "batch_size": batch_size,
            "input_tokens": int(input_ids.shape[-1]),
            "max_new_tokens": args.max_new_tokens,
            "elapsed_seconds": elapsed,
            "model_load_seconds": load_seconds,
            "gpu_memory_used_mib_after": int(mem.used / 1024 / 1024),
            "pcie_link_gen": sampler.link_gen,
            "pcie_link_width": sampler.link_width,
            "pcie_samples": pcie.samples,
            "pcie_rx_avg_kbps": pcie.avg_rx_kbps,
            "pcie_tx_avg_kbps": pcie.avg_tx_kbps,
            "pcie_rx_max_kbps": pcie.max_rx_kbps,
            "pcie_tx_max_kbps": pcie.max_tx_kbps,
            "pcie_rx_approx_mb": pcie.approx_rx_mb,
            "pcie_tx_approx_mb": pcie.approx_tx_mb,
            "pcie_rx_util_pct_avg": pcie.rx_util_pct_avg,
            "pcie_tx_util_pct_avg": pcie.tx_util_pct_avg,
            "pcie_rx_util_pct_max": pcie.rx_util_pct_max,
            "pcie_tx_util_pct_max": pcie.tx_util_pct_max,
            "generated_tail": generated_tail,
            **metric_summary,
            **dispatcher_summary,
        }
        rows.append(row)
        print("metric_row=" + json.dumps(row, default=str), flush=True)

    write_outputs(args.output_dir, rows)


if __name__ == "__main__":
    main()
