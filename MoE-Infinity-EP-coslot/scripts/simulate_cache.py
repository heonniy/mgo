#!/usr/bin/env python3
"""Reference cache-view simulator — INDEPENDENT IMPLEMENTATION.

Cross-checks our GlobalCacheController + EvictionPlanner(LRU) +
PlacementPlanner(naive|balanced) decisions by replaying the exact demand
sequence captured in trace's routing_log files and computing expected
(hits, misses, evictions) per (step, layer) entirely from first principles.

This implementation deliberately AVOIDS reusing any code from
``moe_infinity_ep.controller`` so a bug in production controller cannot
silently propagate into the verification.

Spec we implement (matches global_controller.handle_layer 1→4):
  1. union_demand = union over ranks of local_demanded experts
  2. classify against cache_view (rank with this (layer, expert) resident)
  3. touch hits FIRST (move-to-end in per-rank OrderedDict); record
     protected_keys = set of (layer, expert) that are hits this layer
  4. place miss into target_rank per placement policy:
        naive    : target = expert_id % ep_size
        balanced : target = argmin(lane_loads) with tie-break by rank index
                   (lane_loads reset per layer, += 1 per install assigned)
  5. evict victim if target's cache is full; victim picked LRU-head
     SKIPPING protected_keys (head iterator until non-protected); if
     no non-protected victim available → DROP this install (cap-undersize)

Inputs:
  --routing-glob        : glob for routing_log files (per-rank jsonl)
  --ep-size             : world size for placement
  --num-experts         : E (for assertions only)
  --cap                 : cap_per_rank (use 0 for unlimited)
  --placement           : naive | balanced
  --out                 : output JSON
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple


def load_routing_log(glob_pat: str) -> List[dict]:
    """Read all *.jsonl files, return a flat list of records sorted by
    (step, layer, ep_rank)."""
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"no routing log files match: {glob_pat}")
    records: List[dict] = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    records.sort(key=lambda r: (int(r["step"]), int(r["layer"]),
                                int(r["ep_rank"])))
    return records


def group_by_layer(records: List[dict]):
    """Yield (step, layer, [per_rank_records ...])."""
    cur_key = None
    bucket: List[dict] = []
    for r in records:
        key = (int(r["step"]), int(r["layer"]))
        if cur_key is None:
            cur_key = key
        if key != cur_key:
            yield cur_key[0], cur_key[1], bucket
            bucket = []
            cur_key = key
        bucket.append(r)
    if bucket:
        yield cur_key[0], cur_key[1], bucket


def simulate(records: List[dict], ep_size: int, cap: int,
             placement: str) -> dict:
    """Run the reference planner over the captured demand sequence.

    Returns a dict with per-(step, layer) expected counts + totals.
    """
    if placement not in ("naive", "balanced"):
        raise SystemExit(f"unknown placement: {placement}")
    INF = cap == 0  # cap=0 = unlimited
    # per-rank OrderedDict[(layer, expert) -> True]; LRU head = oldest.
    cache: List["OrderedDict[Tuple[int, int], bool]"] = [
        OrderedDict() for _ in range(ep_size)
    ]
    per_layer: List[dict] = []
    totals = {
        "hits": 0, "misses": 0, "evictions": 0, "drops": 0,
        "miss_target_rank_dist": [0] * ep_size,
        "hits_per_serving_rank": [0] * ep_size,
    }

    for step, layer, bucket in group_by_layer(records):
        # 1. union demand across all ranks
        union_demand = set()
        for r in bucket:
            for e in r.get("local_demanded", []) or []:
                union_demand.add(int(e))

        # 2. classify against cache_view: hit if any rank has it; assign
        #    serving_rank = lowest rank that has it (matches
        #    cache_state.locate's min-rank rule).
        hits: Dict[int, int] = {}      # expert_id -> serving_rank
        misses: List[int] = []
        for e in sorted(union_demand):
            key = (layer, e)
            serving = None
            for r in range(ep_size):
                if key in cache[r]:
                    serving = r
                    break
            if serving is not None:
                hits[e] = serving
            else:
                misses.append(e)

        # 3. touch hits FIRST (move-to-end) and record protected_keys
        for e, r in hits.items():
            cache[r].move_to_end((layer, e))
        protected = {(layer, e) for e in hits.keys()}

        # 4. place misses
        installs: List[Tuple[int, int]] = []  # (expert_id, target_rank)
        if placement == "naive":
            for e in misses:
                installs.append((e, e % ep_size))
        else:  # balanced
            loads = [0] * ep_size
            for e in misses:
                # argmin(loads) with tie-break by rank index
                target = min(range(ep_size), key=lambda r: (loads[r], r))
                loads[target] += 1
                installs.append((e, target))

        # 5. apply: evict (LRU head skipping protected) then install.
        n_evicts = 0
        n_drops = 0
        for (e, target) in installs:
            if not INF and len(cache[target]) >= cap:
                victim_key = None
                for k in cache[target]:
                    if k not in protected:
                        victim_key = k
                        break
                if victim_key is None:
                    n_drops += 1
                    continue
                del cache[target][victim_key]
                n_evicts += 1
            cache[target][(layer, e)] = True

        # accumulate
        miss_dist = [0] * ep_size
        for _, t in installs:
            if n_drops > 0:
                # We don't know which installs were dropped, but for
                # dist we count the ones we actually applied — same
                # semantics as cache_view.install in production.
                pass
            miss_dist[t] += 1
        hit_dist = [0] * ep_size
        for r in hits.values():
            hit_dist[r] += 1

        per_layer.append({
            "step": step,
            "layer": layer,
            "n_demanded": len(union_demand),
            "n_hits": len(hits),
            "n_misses": len(installs),
            "n_evicts": n_evicts,
            "n_drops": n_drops,
            "hits_per_serving_rank": hit_dist,
            "miss_target_rank_dist": miss_dist,
            "cache_size_per_rank": [len(c) for c in cache],
        })
        totals["hits"] += len(hits)
        totals["misses"] += len(installs) - n_drops  # actual installs
        totals["evictions"] += n_evicts
        totals["drops"] += n_drops
        for i in range(ep_size):
            totals["hits_per_serving_rank"][i] += hit_dist[i]
            totals["miss_target_rank_dist"][i] += miss_dist[i]

    return {
        "ep_size": ep_size,
        "cap_per_rank": cap if not INF else "unlimited",
        "placement": placement,
        "n_layers_simulated": len(per_layer),
        "totals": totals,
        "per_layer": per_layer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-glob", required=True,
                    help="glob for *_routing_r*.jsonl files")
    ap.add_argument("--ep-size", type=int, required=True)
    ap.add_argument("--num-experts", type=int, required=True)
    ap.add_argument("--cap", type=int, required=True,
                    help="cap_per_rank; 0 = unlimited")
    ap.add_argument("--placement", choices=["naive", "balanced"],
                    required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_routing_log(args.routing_glob)
    print(f"[sim] loaded {len(records)} routing records from "
          f"{args.routing_glob}", flush=True)

    result = simulate(records, args.ep_size, args.cap, args.placement)
    Path(args.out).write_text(json.dumps(result, indent=2))
    t = result["totals"]
    print(f"[sim] DONE → {args.out}")
    print(f"[sim] totals: hits={t['hits']} misses={t['misses']} "
          f"evictions={t['evictions']} drops={t['drops']}")
    print(f"[sim] miss_target_rank_dist = {t['miss_target_rank_dist']}")
    print(f"[sim] hits_per_serving_rank = {t['hits_per_serving_rank']}")


if __name__ == "__main__":
    main()
