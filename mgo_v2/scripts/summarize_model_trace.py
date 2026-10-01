#!/usr/bin/env python3
"""Join Nsight per-cell GPU activity to all-rank physical fetch receipts."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import sqlite3

from summarize_slot_trace import merge


def collect_activity(db, ranges, expert_bytes):
    """Scan each large activity table once, keeping only measured NCCL intervals."""
    by_pid, starts, totals = {}, {}, {}
    for start, end, tid, label in ranges:
        pid = tid & ~((1 << 24) - 1)  # low 24 globalTid bits encode the thread
        by_pid.setdefault(pid, []).append((start, end, label))
        totals[label] = {"nccl": [], "names": set(), "expert_h2d": 0,
                         "all_h2d": 0, "sizes": set()}
    for pid, windows in by_pid.items():
        windows.sort()
        if any(a[1] > b[0] for a, b in zip(windows, windows[1:])):
            raise ValueError("measured cell ranges overlap within a worker")
        starts[pid] = [window[0] for window in windows]

    def target(pid, start, end):
        index = bisect_right(starts.get(pid, []), start) - 1
        if index < 0:
            return None
        window = by_pid[pid][index]
        return totals[window[2]] if end <= window[1] else None

    for pid, start, end, name in db.execute(
            "SELECT k.globalPid,k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName=s.id WHERE lower(s.value) LIKE '%nccl%'"):
        row = target(pid, start, end)
        if row is not None:
            row["nccl"].append((start, end))
            row["names"].add(name)
    for pid, start, end, size in db.execute(
            "SELECT globalPid,start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1"):
        row = target(pid, start, end)
        if row is not None:
            row["all_h2d"] += size
            # Also catch the old three-projection CPU-view upload bypass.
            if size >= expert_bytes // 3:
                row["expert_h2d"] += size
                row["sizes"].add(size)
    return totals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", required=True)
    p.add_argument("--receipts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", help="optional unprofiled matrix receipt directory for semantic parity")
    p.add_argument("--expert-bytes", type=int, default=9437184)
    args = p.parse_args()
    db = sqlite3.connect(f"file:{Path(args.sqlite).resolve()}?mode=ro", uri=True)
    ranges = list(db.execute("SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text LIKE 'mgo_cell:%'"))
    if not ranges:
        raise ValueError("no measured mgo_cell NVTX ranges in trace")
    receipts = Path(args.receipts)
    expected_files = list(receipts.glob("*-rep*-rank*.json"))
    expected_labels = set()
    rank_counts = {}
    for path in expected_files:
        name, suffix = path.stem.rsplit("-rep", 1)
        repeat, rank = suffix.split("-rank")
        expected_labels.add(f"mgo_cell:{rank}:{name}:{repeat}")
        rank_counts[int(rank)] = rank_counts.get(int(rank), 0) + 1
    labels = [row[3] for row in ranges]
    if len(set(labels)) != len(labels) or set(labels) != expected_labels:
        raise AssertionError("trace ranges do not cover exactly the completed rank receipts")
    for rank, count in rank_counts.items():
        completed = json.loads((receipts / f"nsight-rank{rank}.json").read_text())
        assert completed["rank"] == rank and completed["measured_cells"] == count
        if args.baseline:
            recorded = json.loads((receipts / f"provenance-rank{rank}.json").read_text())
            original = json.loads((Path(args.baseline) / f"provenance-rank{rank}.json").read_text())
            for key in ("rank", "world", "steps", "checkpoint", "code_sha256", "versions"):
                assert recorded[key] == original[key], f"rank {rank}: {key} provenance drift"
            for key in ("similarity", "affinity", "workload"):
                assert recorded["input_sha256"][key] == original["input_sha256"][key]
    activity = collect_activity(db, ranges, args.expert_bytes)
    results = []
    for start, end, tid, label in ranges:
        _, rank, name, repeat = label.split(":")
        measured = activity[label]
        nccl = measured["nccl"]
        receipt = json.loads((receipts / f"{name}-rep{repeat}-rank{rank}.json").read_text())
        if args.baseline:
            baseline = json.loads((Path(args.baseline) / f"{name}-rep{repeat}-rank{rank}.json").read_text())
            assert receipt["config"] == baseline["config"], label + ": configuration drift"
            assert receipt["metrics"] == baseline["metrics"], label + ": policy drift under profiling"
            assert receipt["quality"] == baseline["quality"], label + ": output drift under profiling"
        actual = measured["expert_h2d"]
        expected = receipt["host_fetch_bytes"]
        result = {"rank": int(rank), "cell": name, "repeat": int(repeat),
                  "generation_range_ms": (end - start) / 1e6,
                  "nccl_kernel_count": len(nccl),
                  "nccl_kernel_sum_ms": sum(b - a for a, b in nccl) / 1e6,
                  "nccl_kernel_union_ms": sum(b - a for a, b in merge(nccl)) / 1e6,
                  "expert_h2d_bytes": actual, "logical_fetch_bytes": expected,
                  "large_h2d_copy_sizes": sorted(measured["sizes"]),
                  "expert_h2d_matches_logical": actual == expected,
                  "all_h2d_bytes": measured["all_h2d"],
                  "nccl_kernel_names": sorted(measured["names"])}
        if not nccl or actual != expected:
            raise AssertionError(f"incomplete trace or mismatched transfers: {result}")
        results.append(result)
    if len(results) != len(expected_files):
        raise AssertionError(f"trace has {len(results)} cells, receipts have {len(expected_files)}")
    summary = {"status": "PASS", "trace": str(Path(args.sqlite).resolve()),
               "baseline_semantics_verified": bool(args.baseline),
               "rank_cell_count": len(results), "cells": results,
               "scope": "Nsight diagnostic kernel durations, including device-side peer waits; not link latency or uninstrumented model timings. NCCL tensor bytes are in benchmark receipts, not wire-byte counters."}
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}))


if __name__ == "__main__":
    main()
