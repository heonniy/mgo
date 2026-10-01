"""Phase 2 — production controller vs independent oracle (differential).

Drives the REAL GlobalCacheController and the reference oracle through the same
deterministic adversarial traces and asserts EVERY field matches each layer:
demand, hit_ops, miss_per_rank, misses, fetch_ops (order preserved), routing_map,
shadow_after.  Plus structural invariants (B*/accounting).

Ground truth = the oracle (reference.py), NOT the suspect golden JSONL.

Run:  pytest tests/ref/test_controller_vs_ref.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reference import RefController, demand_matrix          # noqa: E402
import prod_driver as pd                                     # noqa: E402

L = 0
FIELDS = ("demand", "hit_ops", "miss_per_rank", "misses",
          "fetch_ops", "routing_map", "shadow_after")


# ------------------------------------------------------------------
# adversarial traces: (name, ep, cap, num_experts, owner_hint, [step dicts])
# each step dict is {rank: {expert: count}} (demander rank is irrelevant to
# planning; only the global per-expert sum and which experts are demanded
# matter — but we exercise multi-rank demand to test union dedup).
# ------------------------------------------------------------------
def _T_divisible():
    # ep=2, cap=2: 4 distinct misses over 2 ranks -> even split each step.
    return ("divisible", 2, 2, 8,
            [{0: {0: 5, 1: 4, 2: 3, 3: 2}},     # 4 miss -> rank0{0,2} rank1{1,3}
             {0: {0: 9, 2: 1}},                  # hits (resident) -> no fetch
             {0: {4: 6, 5: 5, 6: 4, 7: 3}}])     # 4 new miss -> evictions


def _T_indivisible():
    # ep=2, cap=3: 3 misses -> 2/1 split (max-min=1 for balanced).
    return ("indivisible", 2, 3, 8,
            [{0: {0: 7, 1: 6, 2: 5}},
             {0: {3: 4, 4: 3, 5: 2, 6: 1}},
             {0: {0: 9, 3: 8, 6: 7}}])


def _T_naive_pileup():
    # ep=4: experts all == 0 mod 4 -> naive piles on rank0, balanced spreads.
    return ("naive_pileup", 4, 4, 32,
            [{0: {0: 8, 4: 7, 8: 6, 12: 5}},
             {0: {16: 4, 20: 3, 24: 2, 28: 1}},
             {0: {0: 9, 16: 8}}])


def _T_dup_demand():
    # same expert demanded by several ranks -> union dedup -> single fetch.
    return ("dup_demand", 4, 4, 16,
            [{0: {5: 3}, 1: {5: 4}, 2: {5: 2}, 3: {5: 1}},   # all want E5
             {0: {5: 2}, 1: {6: 9}, 2: {6: 1}}])             # E5 hit, E6 dup miss


def _T_capacity_pressure():
    # ep=2, cap=2, many distinct experts -> sustained eviction pressure.
    steps = []
    for s in range(6):
        base = (s * 2) % 8
        steps.append({0: {base: 7 - s, (base + 1) % 8: 3, (base + 4) % 8: 2}})
    return ("cap_pressure", 2, 2, 8, steps)


def _T_single_rank():
    # ep=1 path (router-mask stub) — still must match the oracle.
    return ("single_rank", 1, 3, 8,
            [{0: {0: 5, 1: 4, 2: 3}},
             {0: {0: 9}},
             {0: {3: 6, 4: 5, 5: 4}},
             {0: {3: 2, 6: 1}}])


TRACES = [_T_divisible(), _T_indivisible(), _T_naive_pileup(),
          _T_dup_demand(), _T_capacity_pressure(), _T_single_rank()]

OWNERS = ["naive", "balanced"]
EVICTS = ["lru", "lfu"]


def _ids(seq):
    return [t[0] for t in seq]


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("evict", EVICTS)
@pytest.mark.parametrize("trace", TRACES, ids=_ids(TRACES))
def test_prod_matches_oracle(owner, evict, trace):
    name, ep, cap, ne, steps = trace
    prod = pd.make_prod(ep, cap, ne, owner, evict)
    ref = RefController(ep, cap, ne, owner=owner, evict=evict)

    for si, step in enumerate(steps):
        matrix = demand_matrix(ep, ne, step)
        a = pd.prod_step(prod, matrix, L)
        b = ref.step(matrix, L)
        for f in FIELDS:
            assert a[f] == b[f], (
                f"[{name} owner={owner} evict={evict} step={si}] "
                f"field '{f}' mismatch:\n  prod={a[f]}\n  ref ={b[f]}")
        # accounting invariants on the agreed output
        gud = len(a["demand"])
        hit = len(a["hit_ops"])
        miss = len(a["misses"])
        fetch = len(a["fetch_ops"])
        evicts = sum(1 for op in a["fetch_ops"]
                     if op["victim_expert"] is not None)
        assert gud == hit + miss
        assert miss == fetch                          # B9 / dedup
        # no duplicate fetch
        seen = [tuple(op["expert"]) for op in a["fetch_ops"]]
        assert len(seen) == len(set(seen)), f"duplicate fetch in {name}"
        # no expert resident in >1 (rank,slot)
        res = [c for r in a["shadow_after"].values() for c in r if c]
        assert len(res) == len(set(map(tuple, res))), f"dup resident in {name}"
        # evictions only when cap full and miss>free
        assert evicts <= fetch

        pd.prod_bump(prod)
        ref.bump()


if __name__ == "__main__":
    for tr in TRACES:
        for o in OWNERS:
            for e in EVICTS:
                test_prod_matches_oracle(o, e, tr)
    print("controller-vs-oracle differential: ALL PASS")
