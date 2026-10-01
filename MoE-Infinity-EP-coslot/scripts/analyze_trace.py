"""Read one or more trace JSON files and print cache + timing summary."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def fmt_us(us: int) -> str:
    if us >= 1_000_000:
        return f"{us/1e6:.2f}s"
    if us >= 1_000:
        return f"{us/1e3:.1f}ms"
    return f"{us}us"


def fmt_bytes(b: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    for u in units:
        if b < 1024:
            return f"{b:.2f}{u}"
        b /= 1024
    return f"{b:.2f}TB"


def analyze_one(trace_path: Path) -> None:
    trace = json.loads(trace_path.read_text())
    totals = trace["totals"]
    per_rank = trace["per_rank"]
    cfg = trace.get("config", {})

    hits = totals.get("cache_hits", 0)
    misses = totals.get("cache_misses", 0)
    evicts = totals.get("cache_evictions", 0)
    layers = totals.get("layers_executed", 0)
    hit_rate = hits / max(1, hits + misses)

    print(f"\n=== {trace_path.name} ===")
    print(
        f"  policy  : fetch={cfg.get('policies', {}).get('fetch_dispatch')} "
        f"placement={cfg.get('policies', {}).get('placement')} "
        f"prefetch={cfg.get('execution', {}).get('enable_prefetch')} "
        f"cap={cfg.get('offload', {}).get('cache_capacity_per_rank') or 'unlimited'}"
    )
    print(f"  ranks   : {trace['world_size']}  ep_size={trace['ep_size']}")
    print(
        f"  cache   : hits={hits} misses={misses} hit_rate={hit_rate*100:.1f}% "
        f"evictions={evicts} remote_serves={totals.get('cache_remote_serves', 0)}"
    )
    print(
        f"  layers  : {layers} executed  tokens={totals.get('tokens_processed', 0)}"
    )
    print(
        f"  nvlink  : activation={fmt_bytes(totals.get('nvlink_activation_bytes', 0))} "
        f"migration={fmt_bytes(totals.get('nvlink_expert_migration_bytes', 0))}"
    )
    print(f"  pcie    : fetch={fmt_bytes(totals.get('pcie_fetch_bytes', 0))}")
    print(
        f"  timing  : total={fmt_us(totals.get('layer_total_us_sum', 0))} "
        f"cache_sync={fmt_us(totals.get('cache_sync_us_sum', 0))} "
        f"a2a_fwd={fmt_us(totals.get('a2a_forward_us_sum', 0))} "
        f"local_exec={fmt_us(totals.get('local_exec_us_sum', 0))} "
        f"a2a_bwd={fmt_us(totals.get('a2a_backward_us_sum', 0))}"
    )

    # Per-rank load balance.
    print(f"  per-rank balance:")
    rank_tokens = [d.get("tokens_processed", 0) for d in per_rank]
    rank_misses = [d.get("cache_misses", 0) for d in per_rank]
    rank_pcie = [d.get("pcie_fetch_bytes", 0) for d in per_rank]
    rank_a2a_us = [d.get("a2a_forward_us_sum", 0) for d in per_rank]
    rank_exec_us = [d.get("local_exec_us_sum", 0) for d in per_rank]
    for i, d in enumerate(per_rank):
        print(
            f"    rank{i}: tokens={d['tokens_processed']} misses={d['cache_misses']} "
            f"hits={d['cache_hits']} evict={d['cache_evictions']} "
            f"local_exec={fmt_us(d['local_exec_us_sum'])} "
            f"a2a_fwd={fmt_us(d['a2a_forward_us_sum'])}"
        )

    # Imbalance summary.
    if len(per_rank) > 1:
        def imbal(xs):
            mn, mx = min(xs), max(xs)
            if mx == 0:
                return 0.0
            return (mx - mn) / mx
        print(
            f"  imbalance: misses={imbal(rank_misses)*100:.1f}% "
            f"exec_time={imbal(rank_exec_us)*100:.1f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", help="trace JSON files to compare")
    args = parser.parse_args()
    for p in args.traces:
        analyze_one(Path(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
