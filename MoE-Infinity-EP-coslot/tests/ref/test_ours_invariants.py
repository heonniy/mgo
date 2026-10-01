"""OURS — algorithm.md §7 invariants over seeded fuzz traces.

For every layer (prefill + decode):
  * per-GPU fetch count <= K = ceil(M/G)                       (§7.1 / §7.2)
  * each miss expert fetched exactly once; total fetched == M  (§7.1 / §7.3)
  * per-rank dst_slots distinct, order contiguous 0..n-1       (§7.4)
  * same-rank fetch order is D_now / D_prefill descending      (§1.8 / §4)
  * no expert resident in >1 (rank, slot) in shadow_after
  * fetched experts are disjoint from this layer's hits

Traces are sized so the whole expert set fits (ep*cap >= ne) -> no
over-subscription drop, so total fetched == M exactly.

Run:  pytest tests/ref/test_ours_invariants.py -q
"""
from __future__ import annotations

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reference import demand_matrix                      # noqa: E402
import prod_driver as pd                                  # noqa: E402

L = 0


def _gen_trace(rng):
    ep = rng.choice([1, 2, 4])
    ne = rng.choice([8, 12, 16])
    cap = max(2, math.ceil(ne / ep))     # ep*cap >= ne -> always fits
    n_steps = rng.randint(3, 8)
    steps = []
    for _ in range(n_steps):
        k = rng.randint(1, min(ne, 10))
        step = {r: {} for r in range(ep)}
        for e in rng.sample(range(ne), k):
            step[rng.randrange(ep)][e] = rng.randint(1, 20)
        steps.append(step)
    return ep, cap, ne, steps


def _check_layer(ep, ne, matrix, a, is_decode, ctx):
    misses = a["misses"]
    M = len(misses)
    K = math.ceil(M / ep) if M else 0
    fetch_ops = a["fetch_ops"]

    # total fetched == M (no over-subscription in these traces), each once
    fetched = [op["expert"] for op in fetch_ops]
    assert len(fetched) == M, f"{ctx} fetched {len(fetched)} != M {M}"
    assert len(set(fetched)) == M, f"{ctx} duplicate fetch"
    assert set(fetched) == set(misses), f"{ctx} fetched set != miss set"

    # per-rank: fetch count <= K, dst distinct, order contiguous, D desc
    for r in range(ep):
        ops_r = [op for op in fetch_ops if op["fetcher_rank"] == r]
        assert len(ops_r) <= K, f"{ctx} rank {r} fetch {len(ops_r)} > K {K}"
        dsts = [op["dst_slot"] for op in ops_r]
        assert len(dsts) == len(set(dsts)), f"{ctx} rank {r} dup dst_slot"
        ordered = sorted(ops_r, key=lambda o: o["order"])
        assert [o["order"] for o in ordered] == list(range(len(ops_r))), (
            f"{ctx} rank {r} order not contiguous")
        d_seq = [matrix[r][op["expert"][1]] for op in ordered]
        assert all(d_seq[i] >= d_seq[i + 1] for i in range(len(d_seq) - 1)), (
            f"{ctx} rank {r} fetch order not D desc: {d_seq}")

    # no expert resident in >1 (rank, slot)
    res = [c for slots in a["shadow_after"].values() for c in slots if c]
    assert len(res) == len(set(res)), f"{ctx} duplicate resident"

    # hits disjoint from fetched
    hit_experts = {op["expert"] for op in a["hit_ops"]}
    assert hit_experts.isdisjoint(set(fetched)), f"{ctx} hit/miss overlap"


@pytest.mark.parametrize("is_decode", [False, True], ids=["prefill", "decode"])
@pytest.mark.parametrize("seed", range(150))
def test_ours_invariants(seed, is_decode):
    rng = random.Random(seed * 6271 + 3)
    ep, cap, ne, steps = _gen_trace(rng)
    prod = pd.make_prod_ours(ep, cap, ne)
    for si, step in enumerate(steps):
        matrix = demand_matrix(ep, ne, step)
        a = pd.prod_step(prod, matrix, L, is_decode=is_decode)
        _check_layer(ep, ne, matrix, a, is_decode,
                     f"[seed={seed} decode={is_decode} step={si}]")
        pd.prod_bump(prod)


def test_ours_decode_oversubscription_drop():
    """cap*ep < per-step misses -> planner drops lowest-demand misses, still
    produces a valid (drift-free) plan for what it fetches."""
    ep, cap, ne = 2, 2, 16          # capacity 4 < up-to-16 misses
    prod = pd.make_prod_ours(ep, cap, ne)
    step = {0: {e: (e + 1) for e in range(8)}, 1: {e: 1 for e in range(8, 16)}}
    matrix = demand_matrix(ep, ne, step)
    a = pd.prod_step(prod, matrix, L, is_decode=True)
    # fetched at most total physical capacity
    assert len(a["fetch_ops"]) <= ep * cap
    # highest-demand experts (7,6,5,...) survive the drop
    fetched_ids = {op["expert"][1] for op in a["fetch_ops"]}
    assert 7 in fetched_ids
    # shadow has no duplicate residents
    res = [c for slots in a["shadow_after"].values() for c in slots if c]
    assert len(res) == len(set(res))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
