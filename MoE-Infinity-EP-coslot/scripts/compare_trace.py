#!/usr/bin/env python3
"""Compare reference simulator output vs actual trace.

Inputs:
  --simulation  : path to simulate_cache.py output JSON
  --routing-glob: same glob used to feed the simulator (we re-aggregate
                  actual per-(step, layer) counts from routing_log entries
                  for an apples-to-apples comparison)
  --strict-rank-dist : also compare miss_target_rank_dist / hits_per_serving_rank

Diff rules:
  * Per (step, layer) compare: n_demanded, n_hits, n_misses, n_evicts.
  * If any mismatch, record (step, layer, field, expected, actual).
  * Print first 20 diffs + totals.
  * Exit 0 if no diff; 1 otherwise.

Cross-check semantics (what each field means):
  - simulator's n_demanded   == |union_demand_layer|              (independent)
  - simulator's n_hits       == hits in reference cache_view
  - simulator's n_misses     == install_ops applied (after drop)
  - simulator's n_evicts     == evictions performed
  - actual (from routing_log): aggregated across ranks for same (step, layer)
                               summing across ranks where applicable

Equality means our controller's classify/place/evict on this seed produces
the same decisions as the reference. Differences indicate a controller bug
OR a reference-simulator bug — first 20 diffs help triangulate.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_actual(glob_pat: str) -> Dict[Tuple[int, int], dict]:
    """Aggregate routing_log records by (step, layer).

    Each layer has one entry per rank (sum of demand, hits, misses, evicts
    is taken as the global count).  Hits are union of expert keys but
    individual rank logs only record the count under the rank's view —
    since the controller uses a byte-identical cache_view, n_hits should
    be the SAME on every rank.  We take the max (defensive: should equal
    min).
    """
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"no routing log files match: {glob_pat}")
    per_layer: Dict[Tuple[int, int], dict] = defaultdict(lambda: {
        "n_hits_values": [],
        "n_misses_values": [],
        "n_evicts_values": [],
        "local_demanded_union": set(),
        "miss_target_rank_dist": None,
        "hits_per_serving_rank": None,
    })
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (int(r["step"]), int(r["layer"]))
                acc = per_layer[key]
                acc["n_hits_values"].append(int(r["n_hits"]))
                acc["n_misses_values"].append(int(r["n_misses"]))
                acc["n_evicts_values"].append(int(r["n_evicts"]))
                for e in r.get("local_demanded", []) or []:
                    acc["local_demanded_union"].add(int(e))
                # rank-symmetric: every rank's view of these dists should
                # match (decisions are byte-identical). Capture first.
                if acc["miss_target_rank_dist"] is None:
                    acc["miss_target_rank_dist"] = list(
                        r.get("miss_target_rank_dist", []))
                    acc["hits_per_serving_rank"] = list(
                        r.get("hits_per_serving_rank", []))
    # Reduce
    out: Dict[Tuple[int, int], dict] = {}
    for key, acc in per_layer.items():
        out[key] = {
            "n_demanded": len(acc["local_demanded_union"]),
            "n_hits": max(acc["n_hits_values"] or [0]),
            "n_misses": max(acc["n_misses_values"] or [0]),
            "n_evicts": max(acc["n_evicts_values"] or [0]),
            "miss_target_rank_dist": acc["miss_target_rank_dist"] or [],
            "hits_per_serving_rank": acc["hits_per_serving_rank"] or [],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulation", required=True)
    ap.add_argument("--routing-glob", required=True)
    ap.add_argument("--strict-rank-dist", action="store_true",
                    help="also compare per-rank miss/hit distribution")
    ap.add_argument("--max-diffs", type=int, default=20)
    args = ap.parse_args()

    sim = json.loads(Path(args.simulation).read_text())
    actual = load_actual(args.routing_glob)

    # Build expected map: (step, layer) -> dict.
    expected: Dict[Tuple[int, int], dict] = {}
    for r in sim["per_layer"]:
        expected[(int(r["step"]), int(r["layer"]))] = r

    diffs: List[dict] = []
    keys_sim_only = set(expected) - set(actual)
    keys_act_only = set(actual) - set(expected)
    common = sorted(set(expected) & set(actual))

    for key in common:
        e = expected[key]
        a = actual[key]
        for field in ("n_demanded", "n_hits", "n_misses", "n_evicts"):
            if int(e[field]) != int(a[field]):
                diffs.append({
                    "step": key[0], "layer": key[1], "field": field,
                    "expected": int(e[field]), "actual": int(a[field]),
                })
        if args.strict_rank_dist:
            for field in ("miss_target_rank_dist", "hits_per_serving_rank"):
                if list(e[field]) != list(a[field]):
                    diffs.append({
                        "step": key[0], "layer": key[1], "field": field,
                        "expected": list(e[field]),
                        "actual": list(a[field]),
                    })

    # Totals comparison
    sim_totals = sim["totals"]
    act_totals = {
        "hits":      sum(a["n_hits"]    for a in actual.values()),
        "misses":    sum(a["n_misses"]  for a in actual.values()),
        "evictions": sum(a["n_evicts"]  for a in actual.values()),
    }

    print("=" * 60)
    print(f"compare_trace: sim={args.simulation}")
    print(f"               actual={args.routing_glob}")
    print(f"layers_compared = {len(common)}")
    print(f"sim_only_layers = {len(keys_sim_only)}  "
          f"actual_only_layers = {len(keys_act_only)}")
    print(f"TOTALS (sim vs actual):")
    print(f"  hits      : sim={sim_totals['hits']:>8d}  "
          f"actual={act_totals['hits']:>8d}  "
          f"Δ={act_totals['hits'] - sim_totals['hits']:+d}")
    print(f"  misses    : sim={sim_totals['misses']:>8d}  "
          f"actual={act_totals['misses']:>8d}  "
          f"Δ={act_totals['misses'] - sim_totals['misses']:+d}")
    print(f"  evictions : sim={sim_totals['evictions']:>8d}  "
          f"actual={act_totals['evictions']:>8d}  "
          f"Δ={act_totals['evictions'] - sim_totals['evictions']:+d}")
    print(f"  drops(sim): {sim_totals['drops']:>8d}")
    print()
    if not diffs and not keys_sim_only and not keys_act_only:
        print("✓ PASS  — every layer's counts match the reference exactly.")
        sys.exit(0)
    print(f"✗ DIFF  — {len(diffs)} field mismatches across "
          f"{len({(d['step'], d['layer']) for d in diffs})} layers.")
    for d in diffs[: args.max_diffs]:
        print(f"  step={d['step']:>3d} L={d['layer']:>3d}  "
              f"{d['field']:>22s}  expected={d['expected']}  "
              f"actual={d['actual']}")
    if len(diffs) > args.max_diffs:
        print(f"  … and {len(diffs) - args.max_diffs} more diffs not shown.")
    sys.exit(1)


if __name__ == "__main__":
    main()
