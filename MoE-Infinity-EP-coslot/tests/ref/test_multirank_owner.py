"""Phase 5 — multi-rank owner balance + queue-hazard negative tests.

Owner-policy correctness across ep_size in {2,4}:
  * naive   : per-rank fetch count == #{miss expert : e % ep == r}  (B5)
  * balanced: per-rank fetch count max-min <= 1 every layer          (B3)
  * balanced reduces (or ties) the imbalance vs naive                (B6)
            strict on the adversarial all-same-residue trace.

Note: in this codebase owner assignment is pure fetch-COUNT balancing,
independent of capacity (capacity only affects WHICH slot, never WHICH rank).
So balanced always achieves max-min<=1 on the misses; there is no quota/PCIe
ceil that breaks it here.  (Byte/compute balance is a different policy —
demand_aware — out of scope.)

Plus negative tests for the queue hazard contract (PlanQueue order gap).

Run:  pytest tests/ref/test_multirank_owner.py -q
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reference import RefArcher, demand_matrix             # noqa: E402
import prod_driver as pd                                     # noqa: E402

L = 0


def _fetch_counts(miss_per_rank, ep):
    return [len(miss_per_rank[r]) for r in range(ep)]


def _imbalance(counts):
    return max(counts) - min(counts)


# ------------------------------------------------------------------
# B3 — balanced owner: max-min <= 1 on the misses, every layer, any ep.
# ------------------------------------------------------------------
@pytest.mark.parametrize("ep", [2, 4])
@pytest.mark.parametrize("seed", range(60))
def test_balanced_max_minus_min_le_1(ep, seed):
    rng = random.Random(seed * 31 + ep)
    ne = rng.choice([16, 24, 32])
    cap = ne  # large cap -> no eviction, isolate owner balance
    prod = pd.make_prod(ep, cap, ne, owner="balanced", evict="lru")
    for _ in range(rng.randint(2, 6)):
        k = rng.randint(1, ne)
        step = {0: {e: rng.randint(1, 9) for e in rng.sample(range(ne), k)}}
        a = pd.prod_step(prod, demand_matrix(ep, ne, step), L)
        counts = _fetch_counts(a["miss_per_rank"], ep)
        assert _imbalance(counts) <= 1, (
            f"balanced imbalance {counts} > 1 (ep={ep} seed={seed})")
        pd.prod_bump(prod)


# ------------------------------------------------------------------
# B5 — naive owner: count[r] == #{miss e : e % ep == r}, exactly.
# ------------------------------------------------------------------
@pytest.mark.parametrize("ep", [2, 4])
@pytest.mark.parametrize("seed", range(60))
def test_naive_matches_modrule(ep, seed):
    rng = random.Random(seed * 17 + ep)
    ne = rng.choice([16, 24, 32])
    cap = ne
    prod = pd.make_prod(ep, cap, ne, owner="naive", evict="lru")
    for _ in range(rng.randint(2, 6)):
        k = rng.randint(1, ne)
        experts = rng.sample(range(ne), k)
        step = {0: {e: rng.randint(1, 9) for e in experts}}
        a = pd.prod_step(prod, demand_matrix(ep, ne, step), L)
        # expected: only NOT-yet-resident experts are misses; among those,
        # count by e % ep.
        miss_experts = [e for (_, e) in a["misses"]]
        expected = [sum(1 for e in miss_experts if e % ep == r)
                    for r in range(ep)]
        actual = _fetch_counts(a["miss_per_rank"], ep)
        assert actual == expected, (
            f"naive count {actual} != mod-rule {expected} (ep={ep})")
        pd.prod_bump(prod)


# ------------------------------------------------------------------
# B6 — adversarial: all misses share residue 0 mod ep.
#   naive piles them ALL on rank0 (imbalance = M); balanced spreads (<=1).
# ------------------------------------------------------------------
@pytest.mark.parametrize("ep", [2, 4])
def test_balanced_beats_naive_on_pileup(ep):
    ne = 8 * ep
    experts = list(range(0, ne, ep))          # all == 0 mod ep
    step = {0: {e: 1 for e in experts}}
    M = len(experts)

    nv = pd.make_prod(ep, ne, ne, owner="naive", evict="lru")
    a_nv = pd.prod_step(nv, demand_matrix(ep, ne, step), L)
    nv_counts = _fetch_counts(a_nv["miss_per_rank"], ep)

    bl = pd.make_prod(ep, ne, ne, owner="balanced", evict="lru")
    a_bl = pd.prod_step(bl, demand_matrix(ep, ne, step), L)
    bl_counts = _fetch_counts(a_bl["miss_per_rank"], ep)

    assert nv_counts[0] == M and _imbalance(nv_counts) == M    # all on rank0
    assert _imbalance(bl_counts) <= 1                          # balanced spreads
    assert _imbalance(bl_counts) < _imbalance(nv_counts)       # strict win


# ------------------------------------------------------------------
# B9 / accounting — per-rank fetch sums equal global fetch == misses.
# ------------------------------------------------------------------
@pytest.mark.parametrize("ep", [2, 4])
def test_per_rank_sum_equals_global(ep):
    ne = 16
    prod = pd.make_prod(ep, 4, ne, owner="balanced", evict="lfu")
    rng = random.Random(99 + ep)
    for _ in range(8):
        step = {0: {e: rng.randint(1, 9)
                    for e in rng.sample(range(ne), rng.randint(1, ne))}}
        a = pd.prod_step(prod, demand_matrix(ep, ne, step), L)
        per_rank = _fetch_counts(a["miss_per_rank"], ep)
        assert sum(per_rank) == len(a["misses"]) == len(a["fetch_ops"])
        pd.prod_bump(prod)


# ==================================================================
# Queue-hazard NEGATIVE tests (H6/H7) — the archer must reject bad plans.
# ==================================================================
def test_planqueue_order_gap_raises():
    """Out-of-order / stale plan id: orders not contiguous 0..n-1 -> raise."""
    a = RefArcher(4)
    a.seed_slots([None, None, None, None])
    raised = False
    try:
        # two misses with orders {0, 2} (missing 1) -> gap
        a.submit_plan(hit_tuples=[],
                      miss_tuples=[(L, 5, 0, -1, -1, 0),
                                   (L, 6, 1, -1, -1, 2)])
    except AssertionError as ex:
        raised = "order gap" in str(ex)
    assert raised, "RefArcher must reject a non-contiguous PlanQueue order"


def test_hit_pretending_raises():
    """Hit tuple points to a slot that does not hold the expert -> raise."""
    a = RefArcher(4)
    a.seed_slots([(L, 0), (L, 1), None, None])
    raised = False
    try:
        a.submit_plan(hit_tuples=[(L, 7, 0)], miss_tuples=[])  # slot0 holds E0
    except AssertionError as ex:
        raised = "HIT MISMATCH" in str(ex)
    assert raised


if __name__ == "__main__":
    for ep in (2, 4):
        for s in range(60):
            test_balanced_max_minus_min_le_1(ep, s)
            test_naive_matches_modrule(ep, s)
        test_balanced_beats_naive_on_pileup(ep)
        test_per_rank_sum_equals_global(ep)
    test_planqueue_order_gap_raises()
    test_hit_pretending_raises()
    print("multi-rank owner + queue-hazard negatives: ALL PASS")
