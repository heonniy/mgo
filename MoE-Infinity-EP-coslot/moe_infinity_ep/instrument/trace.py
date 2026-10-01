"""Trace aggregation — gather per-rank Counters to rank 0, dump JSON + CSV."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .counters import Counters


def gather_and_dump(
    counters: Counters,
    topology,
    out_dir: str,
    config_dict: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """Gather all ranks' counters to global rank 0 and write trace files.

    Returns the trace JSON path on rank 0; None on other ranks.
    """
    import torch.distributed as dist

    world = topology.world_size
    payload = counters.to_dict()
    payload["rank"] = topology.global_rank
    payload["ep_rank"] = topology.ep_rank

    # Routing log: only dumped when MOE_EP_ROUTING_LOG=1 was on AND the user
    # opts in via MOE_EP_DUMP_ROUTING_LOG=1 (otherwise the per-rank object
    # collective payload stays small). Dumped per-rank to {run_id}_routing_rN.jsonl
    # to avoid blowing up the main trace JSON.
    dump_routing = os.environ.get("MOE_EP_DUMP_ROUTING_LOG", "0") == "1"
    routing_log_local = counters.routing_log if dump_routing else None

    # Object collective so we can ship dicts; small payloads only.
    gathered: list = [None] * world if topology.is_rank0 else []
    dist.gather_object(
        payload,
        gathered if topology.is_rank0 else None,
        dst=0,
        group=topology.world_group,
    )

    if not topology.is_rank0:
        # Non-rank-0: write own routing log file locally if requested.
        if dump_routing and routing_log_local:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            rid = run_id or time.strftime("%Y%m%d-%H%M%S")
            rl_path = out / f"{rid}_routing_r{topology.global_rank}.jsonl"
            with rl_path.open("w") as f:
                for rec in routing_log_local:
                    f.write(json.dumps(rec) + "\n")
        return None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    trace_path = out / f"{run_id}.json"

    if dump_routing and routing_log_local:
        rl_path = out / f"{run_id}_routing_r{topology.global_rank}.jsonl"
        with rl_path.open("w") as f:
            for rec in routing_log_local:
                f.write(json.dumps(rec) + "\n")

    totals = _aggregate(gathered)
    trace = {
        "run_id": run_id,
        "world_size": world,
        "ep_size": topology.ep_size,
        "config": config_dict or {},
        "totals": totals,
        "per_rank": gathered,
    }
    trace_path.write_text(json.dumps(trace, indent=2))

    # Also dump per-rank CSV summary for quick eyeballing.
    csv_path = out / f"{run_id}.csv"
    with csv_path.open("w", newline="") as f:
        keys = sorted(k for k in gathered[0].keys() if k not in {"current_step"})
        writer = csv.writer(f)
        writer.writerow(["rank", *keys])
        for d in gathered:
            writer.writerow([d.get("rank")] + [d.get(k) for k in keys])

    return str(trace_path)


def _aggregate(per_rank_payloads: list) -> Dict[str, Any]:
    """Sum-style aggregation over numeric fields + list concat for selected
    per-sample latency lists (TTFT/TPOT/E2E) so the aggregator can compute
    p50/p95 across ranks."""
    if not per_rank_payloads:
        return {}
    totals: Dict[str, Any] = {}
    # List fields we want preserved (concat across ranks) for percentile
    # analysis downstream. Keep deterministic order by sorting.
    LIST_FIELDS_KEEP = {
        "ttft_us", "tpot_us", "e2e_us",
        "prefill_tokens_per_sample", "decode_tokens_per_sample",
        "drift_layer_pre",
    }
    for k, v in per_rank_payloads[0].items():
        if k in {"rank", "ep_rank", "current_step"}:
            continue
        if isinstance(v, (int, float)):
            totals[k] = sum(d.get(k, 0) for d in per_rank_payloads)
        elif k in LIST_FIELDS_KEEP and isinstance(v, list):
            merged: list = []
            for d in per_rank_payloads:
                merged.extend(d.get(k, []) or [])
            totals[k] = sorted(merged)
    return totals
