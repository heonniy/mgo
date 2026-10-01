"""Regenerate trustworthy golden trajectories FROM THE ORACLE.

The previous golden under ``실험/moe_cache_golden/*.jsonl`` is treated as
UNTRUSTED (its answers were judged wrong).  This script writes fresh golden from
the independent oracle (reference.py) — the only thing we trust — into
``tests/ref/golden/``.  Each line carries the full controller trajectory AND the
fake-archer replay (slot_before/after, submit tuples, direct/staging) so the C++
real-archer test (Phase 4) can replay it.

Run:  python tests/ref/regenerate_golden_from_oracle.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from reference import RefController, RefArcher, demand_matrix   # noqa: E402

L = 0
OUT_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _ek(k):
    return None if k is None else [int(k[0]), int(k[1])]


def _shadow_json(shadow):
    return {str(r): [_ek(s) for s in slots] for r, slots in shadow.items()}


# canonical trajectories (config + per-step demand dicts) chosen to exercise
# fill / hit / direct-evict / staging-evict / multi-rank balance.
TRAJECTORIES = [
    {"name": "ep1_cap2_staging", "ep": 1, "cap": 2, "ne": 8,
     "owner": "naive", "evict": "lru",
     "steps": [{0: {0: 5, 1: 4}},                 # fill [E0,E1]
               {0: {0: 6, 1: 3, 2: 2}},           # E0,E1 hits + E2 miss -> STAGING
               {0: {2: 4, 3: 1}}]},               # E2 hit + E3 miss -> direct evict
    {"name": "ep4_cap4_naive_lru", "ep": 4, "cap": 4, "ne": 32,
     "owner": "naive", "evict": "lru",
     "steps": [{0: {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}},
               {0: {0: 9, 4: 8}},                 # hits
               {0: {8: 6, 9: 5, 10: 4, 11: 3}}]},  # new misses
    {"name": "ep2_cap2_balanced_lfu", "ep": 2, "cap": 2, "ne": 8,
     "owner": "balanced", "evict": "lfu",
     "steps": [{0: {0: 10, 1: 1, 2: 9, 3: 2}},
               {0: {0: 5, 2: 5}},
               {0: {4: 7, 5: 6, 6: 1}}]},
]


def gen():
    os.makedirs(OUT_DIR, exist_ok=True)
    ctrl_lines, archer_lines = [], []

    for traj in TRAJECTORIES:
        ep, cap, ne = traj["ep"], traj["cap"], traj["ne"]
        ref = RefController(ep, cap, ne, owner=traj["owner"], evict=traj["evict"])
        physical = [RefArcher(cap) for _ in range(ep)]  # cumulative

        for ts, step in enumerate(traj["steps"]):
            shadow_before = ref.shadow()
            matrix = demand_matrix(ep, ne, step)
            res = ref.step(matrix, L)

            # build per-rank archer submit tuples from the oracle result
            hits = {r: [] for r in range(ep)}
            for op in res["hit_ops"]:
                hits[op["owner_rank"]].append(
                    [L, op["expert"][1], op["slot"]])
            misses = {r: [] for r in range(ep)}
            for op in res["fetch_ops"]:
                v = op["victim_expert"]
                vl, ve = (-1, -1) if v is None else (v[0], v[1])
                misses[op["fetcher_rank"]].append(
                    [L, op["expert"][1], op["dst_slot"], vl, ve, op["order"]])

            # replay into the cumulative physical model
            replay = {}
            for r in range(ep):
                slot_before = physical[r].get_cached_slots()
                physical[r].submit_plan(
                    [tuple(t) for t in hits[r]],
                    [tuple(t) for t in misses[r]])
                replay[str(r)] = {
                    "slot_before": [_ek(s) for s in slot_before],
                    "hit_tuples": hits[r],
                    "miss_tuples": misses[r],
                    "slot_after": [_ek(s) for s in physical[r].get_cached_slots()],
                    "direct": physical[r].direct,
                    "staging": physical[r].staging,
                }
                physical[r].direct = physical[r].staging = 0

            evict_count = sum(1 for op in res["fetch_ops"]
                              if op["victim_expert"] is not None)
            ctrl_lines.append({
                "trace_step": ts,
                "trajectory": traj["name"],
                "config": {"ep_size": ep, "cap_per_rank": cap,
                           "num_experts": ne, "owner_policy": traj["owner"],
                           "evict_policy": traj["evict"]},
                "per_rank_demand": {str(r): {str(e): c for e, c in d.items()}
                                    for r, d in step.items()},
                "expected": {
                    "global_unique_demand": sorted(_ek(k) for k in res["demand"]),
                    "hit_ops": [{"expert": _ek(o["expert"]),
                                 "owner_rank": o["owner_rank"], "slot": o["slot"]}
                                for o in res["hit_ops"]],
                    "misses": [_ek(k) for k in res["misses"]],
                    "miss_per_rank": {str(r): [_ek(k) for k in res["miss_per_rank"][r]]
                                      for r in range(ep)},
                    "fetch_ops": [{"expert": _ek(o["expert"]),
                                   "fetcher_rank": o["fetcher_rank"],
                                   "dst_slot": o["dst_slot"],
                                   "victim": _ek(o["victim_expert"]),
                                   "order": o["order"]}
                                  for o in res["fetch_ops"]],
                    "routing_map": {f"{k[0]}:{k[1]}": v
                                    for k, v in res["routing_map"].items()},
                    "shadow_before": _shadow_json(shadow_before),
                    "shadow_after": _shadow_json(res["shadow_after"]),
                    "summary": {
                        "global_unique_demand_count": len(res["demand"]),
                        "hit_count": len(res["hit_ops"]),
                        "miss_count": len(res["misses"]),
                        "fetch_ops_count": len(res["fetch_ops"]),
                        "evict_ops_count": evict_count,
                    },
                },
                "archer_replay": replay,
            })
            ref.bump()

    with open(os.path.join(OUT_DIR, "controller_gold_trajectory.jsonl"), "w") as f:
        for ln in ctrl_lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")

    # provenance marker
    with open(os.path.join(OUT_DIR, "PROVENANCE.md"), "w") as f:
        f.write("# Golden provenance\n\n"
                "Generated by `tests/ref/regenerate_golden_from_oracle.py` from "
                "the independent oracle `tests/ref/reference.py`.\n\n"
                "The previous golden at `실험/moe_cache_golden/*.jsonl` is "
                "UNTRUSTED (wrong answers) and superseded by this directory.\n")
    print(f"wrote {len(ctrl_lines)} golden lines to {OUT_DIR}")
    return ctrl_lines


if __name__ == "__main__":
    gen()
