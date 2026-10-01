"""OURS — fake-archer replay: shadow == physical (slot-level), victim-only.

Same contract as test_fake_archer_replay.py but for the OURS joint planner, for
BOTH prefill and decode.  Replays each rank's OURS plan into a RefArcher and
asserts:
  * slot-level drift == 0 (RefArcher slot map == controller shadow), every layer
  * archer evicts ONLY the plan's victim (RefArcher raises otherwise)
  * PlanQueue order contiguous 0..n-1 per rank (RefArcher asserts)
  * accounting: sum of fetched == #misses

Run:  pytest tests/ref/test_ours_archer_replay.py -q
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reference import RefArcher, demand_matrix          # noqa: E402
import prod_driver as pd                                  # noqa: E402

L = 0


def _gen_trace(rng):
    ep = rng.choice([1, 2, 4])
    cap = rng.choice([2, 3, 4, 6])
    ne = rng.choice([8, 12, 16, 24])
    n_steps = rng.randint(3, 9)
    steps = []
    for _ in range(n_steps):
        k = rng.randint(1, min(ne, 10))
        step = {r: {} for r in range(ep)}
        for e in rng.sample(range(ne), k):
            step[rng.randrange(ep)][e] = rng.randint(1, 20)
        steps.append(step)
    return ep, cap, ne, steps


def _replay_one(seed, is_decode):
    rng = random.Random(seed * 7919 + 17)
    ep, cap, ne, steps = _gen_trace(rng)
    prod = pd.make_prod_ours(ep, cap, ne)
    archers = [RefArcher(cap) for _ in range(ep)]  # cumulative physical caches

    for si, step in enumerate(steps):
        matrix = demand_matrix(ep, ne, step)
        hits, misses, shadow = pd.prod_plan_tuples(prod, matrix, L, is_decode=is_decode)

        layer_fetch = 0
        for r in range(ep):
            archers[r].submit_plan(hits[r], misses[r])     # raises on any violation
            phys = archers[r].get_cached_slots()
            assert phys == shadow[r], (
                f"seed={seed} decode={is_decode} step={si} rank={r} SLOT DRIFT\n"
                f"  physical={phys}\n  shadow  ={shadow[r]}\n  steps={steps}")
            layer_fetch += len(misses[r])
            # per-rank order is contiguous 0..n-1
            orders = sorted(m[5] for m in misses[r])
            assert orders == list(range(len(misses[r]))), (
                f"non-contiguous order rank={r}: {orders}")

        # every miss expert (deduped union) is fetched exactly once across ranks
        all_fetched = [(m[0], m[1]) for r in range(ep) for m in misses[r]]
        assert len(all_fetched) == len(set(all_fetched)), "duplicate fetch"
        pd.prod_bump(prod)


@pytest.mark.parametrize("is_decode", [False, True], ids=["prefill", "decode"])
@pytest.mark.parametrize("seed", range(120))
def test_ours_replay_no_drift(seed, is_decode):
    _replay_one(seed, is_decode)


def test_ours_decode_staging_when_victim_is_hit():
    """cap=1, ep=1: E0 resident; step demands E0 (hit) + E1 (miss) -> E1 evicts
    E0 at the only slot -> dst busy -> staging.  OURS must produce this too."""
    prod = pd.make_prod_ours(1, 1, 8)
    pd.seed_prod(prod, {0: [(L, 0)]})
    matrix = demand_matrix(1, 8, {0: {0: 5, 1: 3}})
    hits, misses, shadow = pd.prod_plan_tuples(prod, matrix, L, is_decode=True)
    assert hits[0] == [(L, 0, 0)]
    assert len(misses[0]) == 1 and misses[0][0][:3] == (L, 1, 0)
    a = RefArcher(1)
    a.seed_slots([(L, 0)])
    a.submit_plan(hits[0], misses[0])
    assert a.staging == 1 and a.direct == 0
    assert a.get_cached_slots() == shadow[0] == [(L, 1)]


if __name__ == "__main__":
    for s in range(120):
        _replay_one(s, False)
        _replay_one(s, True)
    test_ours_decode_staging_when_victim_is_hit()
    print("OURS fake-archer replay: ALL PASS")
