#!/usr/bin/env python3
"""Extract actual expert transfers and H2D/GEMM overlap from Nsight SQLite."""
import argparse
import json
import re
import sqlite3
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", required=True)
    p.add_argument("--receipt", required=True)
    p.add_argument("--expert-bytes", type=int, default=9437184)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    db = sqlite3.connect(args.sqlite)
    ranges = list(db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='mgo_runtime'"))
    if not ranges:
        raise ValueError("trace has no mgo_runtime ranges")
    def inside(start, end):
        return any(a <= start and end <= b for a, b in ranges)
    copies = [row for row in db.execute("SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY") if inside(row[0],row[1])]
    # Only count full expert or projection transfers, excluding activations
    # and the independent oracle's weight copies outside the runtime ranges.
    expert_copies = [row for row in copies if row[2] >= args.expert_bytes // 3]
    transfers = {}
    for kind, label in ((1, "H2D"), (8, "D2D")):
        selected = [row for row in expert_copies if row[3] == kind]
        transfers[label] = {"copies": len(selected), "bytes": sum(row[2] for row in selected),
                            "summed_ms": sum(row[1] - row[0] for row in selected) / 1e6}
    kernels = [(a,b,name) for a,b,name in db.execute(
        "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id")
        if inside(a,b) and re.search(r"gemm|gemv|cublas|^nvjet_", name, re.I)]
    compute = merge([(a,b) for a,b,_ in kernels])
    overlap = 0
    h2d = [row for row in expert_copies if row[3] == 1]
    for start,end,_,_ in h2d:
        overlap += sum(max(0,min(end,b)-max(start,a)) for a,b in compute)
    receipt = json.loads(Path(args.receipt).read_text())
    expected = receipt["cache_stats"][3] * args.expert_bytes
    result = {"trace": str(Path(args.sqlite).resolve()), "slot_views": receipt["slot_views"],
              "status": receipt["status"], "runtime_ranges": len(ranges), "expert_transfers": transfers,
              "logical_fetch_bytes": expected, "h2d_matches_logical": transfers["H2D"]["bytes"] == expected,
              "h2d_gemm_overlap_ms": overlap / 1e6,
              "h2d_gemm_overlap_fraction": overlap / max(1, sum(b-a for a,b,_,_ in h2d)),
              "matched_gemm_kernels": len(kernels), "gemm_names": sorted(set(name for _,_,name in kernels)),
              "note": "Overlap within this small fixture; not model throughput or a speedup claim."}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k!='gemm_names'}))
    if not result["h2d_matches_logical"]:
        raise AssertionError("actual expert H2D bytes differ from logical misses")
    if result["slot_views"] and transfers["D2D"]["bytes"]:
        raise AssertionError("expert-sized D2D remains with slot views enabled")


if __name__ == "__main__":
    main()
