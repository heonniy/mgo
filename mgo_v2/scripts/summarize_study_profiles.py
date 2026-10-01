#!/usr/bin/env python3
"""Add overlap/activity diagnostics to the validated model trace audit."""
import argparse
from bisect import bisect_right
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from summarize_slot_trace import merge


def duration(intervals):
    return sum(end - start for start, end in merge(intervals))


def intersection(first, second):
    a, b = merge(first), merge(second)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def main():
    parser = argparse.ArgumentParser()
    for name in ("sqlite", "receipts", "baseline", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    audited = root / "profile_transfer_audit.json"
    subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_model_trace.py")),
                    "--sqlite", args.sqlite, "--receipts", args.receipts,
                    "--baseline", args.baseline, "--output", str(audited)], check=True)
    baseline_audit = json.loads(audited.read_text())
    db = sqlite3.connect(f"file:{Path(args.sqlite).resolve()}?mode=ro", uri=True)
    ranges = list(db.execute("SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text LIKE 'mgo_cell:%'"))
    by_pid, windows, activity = {}, {}, {}
    for start, end, tid, label in ranges:
        pid = tid & ~((1 << 24) - 1)
        by_pid.setdefault(pid, []).append((start, end, label))
        activity[label] = dict(all=[], gemm=[], expert_h2d=[], expert_d2d_bytes=0)
        windows[label] = (start, end)
    for values in by_pid.values():
        values.sort()
    starts = {pid: [value[0] for value in values] for pid, values in by_pid.items()}

    def target(pid, start, end):
        index = bisect_right(starts.get(pid, []), start) - 1
        if index < 0:
            return None
        lower, upper, label = by_pid[pid][index]
        return activity[label] if end <= upper else None

    for pid, start, end, name in db.execute("SELECT k.globalPid,k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id"):
        row = target(pid, start, end)
        if row is not None:
            row["all"].append((start, end))
            if any(word in name.lower() for word in ("gemm", "gemv", "cublas", "cutlass", "matmul", "nvjet_")):
                row["gemm"].append((start, end))
    for pid, start, end, size, kind in db.execute("SELECT globalPid,start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        row = target(pid, start, end)
        if row is not None:
            row["all"].append((start, end))
            if kind == 1 and size >= 9437184 // 3:
                row["expert_h2d"].append((start, end))
            if kind == 8 and size >= 9437184:
                row["expert_d2d_bytes"] += size
    # Memset is device activity too; do not mislabel initialization as idle.
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for pid, start, end in db.execute("SELECT globalPid,start,end FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            row = target(pid, start, end)
            if row is not None:
                row["all"].append((start, end))
    ranks = []
    for record in baseline_audit["cells"]:
        assert record["large_h2d_copy_sizes"] == [9437184]
        label = f"mgo_cell:{record['rank']}:{record['cell']}:{record['repeat']}"
        measured = activity[label]
        start, end = windows[label]
        name = f"{record['cell']}-rep{record['repeat']}-rank{record['rank']}.json"
        receipt = json.loads((Path(args.receipts) / name).read_text())
        baseline = json.loads((Path(args.baseline) / name).read_text())
        assert receipt["generated_token_ids"] == baseline["generated_token_ids"]
        assert receipt["cache_stats"] == baseline["cache_stats"]
        assert measured["expert_d2d_bytes"] == 0
        h2d_ns = duration(measured["expert_h2d"])
        overlap_ns = intersection(measured["expert_h2d"], measured["gemm"])
        ranks.append(dict(rank=record["rank"], cell=record["cell"], repeat=record["repeat"],
            actual_expert_h2d_bytes=record["expert_h2d_bytes"], logical_expert_h2d_bytes=record["logical_fetch_bytes"],
            full_expert_d2d_bytes=measured["expert_d2d_bytes"],
            expert_h2d_union_ms=h2d_ns / 1e6, gemm_union_ms=duration(measured["gemm"]) / 1e6,
            h2d_gemm_overlap_ms=overlap_ns / 1e6, h2d_overlap_fraction=overlap_ns / max(h2d_ns, 1),
            nccl_kernel_count=record["nccl_kernel_count"], nccl_kernel_sum_ms=record["nccl_kernel_sum_ms"],
            nccl_kernel_union_ms=record["nccl_kernel_union_ms"], controller_seconds=receipt["controller_seconds"],
            range_ms=(end - start) / 1e6, gpu_activity_union_ms=duration(measured["all"]) / 1e6,
            no_recorded_gpu_activity_ms=((end - start) - duration(measured["all"])) / 1e6))
    rows = []
    for cell in sorted({r["cell"] for r in ranks}):
        records = [r for r in ranks if r["cell"] == cell]
        assert len(records) == 8
        receipts = [json.loads((Path(args.receipts) / f"{cell}-rep0-rank{rank}.json").read_text()) for rank in range(8)]
        kinds = ("dispatch_hidden", "dispatch_metadata", "return_outputs", "return_metadata")
        submitted = sum(sum(r["collectives"]["peer_payload_tx_bytes"].get(k, 0) for k in kinds) for r in receipts)
        metrics = receipts[0]["metrics"]
        derived = metrics["remote_token_rank_pairs"] * (2048 * 2 + 88) + metrics["remote_expert_routes"] * (2048 * 2 + 16)
        assert submitted == derived
        h2d = sum(r["expert_h2d_union_ms"] for r in records)
        overlap = sum(r["h2d_gemm_overlap_ms"] for r in records)
        rows.append(dict(world=8, local_batch=8, cell=cell, repeat=0,
            submitted_peer_payload_tx_bytes=submitted,
            actual_expert_h2d_bytes=sum(r["actual_expert_h2d_bytes"] for r in records),
            logical_expert_h2d_bytes=sum(r["logical_expert_h2d_bytes"] for r in records),
            full_expert_d2d_bytes=sum(r["full_expert_d2d_bytes"] for r in records),
            rank_sum_h2d_union_ms=h2d, rank_sum_h2d_gemm_overlap_ms=overlap,
            h2d_overlap_fraction=overlap / max(h2d, 1e-20),
            nccl_kernel_count=sum(r["nccl_kernel_count"] for r in records),
            max_rank_nccl_kernel_union_ms=max(r["nccl_kernel_union_ms"] for r in records),
            max_rank_controller_seconds=max(r["controller_seconds"] for r in records),
            max_rank_no_recorded_gpu_activity_ms=max(r["no_recorded_gpu_activity_ms"] for r in records)))
    assert len(rows) == 2
    for name, values in (("profile_summary", rows), ("profile_rank_receipts", ranks)):
        with (root / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(values[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
    summary = dict(status="PASS", rows=rows, ranks=ranks,
        scope="Diagnostic only. GEMM includes dense and expert kernels. NCCL kernel intervals include peer waits. Activity gaps are time without recorded kernel/copy/memset, not a measurement of every kind of GPU waiting. No summing overlapping categories into E2E.")
    (root / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", profiles=len(rows), rank_receipts=len(ranks))))


if __name__ == "__main__":
    main()
