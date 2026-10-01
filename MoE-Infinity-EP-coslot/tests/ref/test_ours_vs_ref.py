"""OURS production controller vs independent oracle (differential).

Drives the REAL GlobalCacheController in OURS joint-placement mode and the
independent ``OursRefController`` (re-derived from algorithm.md) through the same
deterministic + seeded-fuzz traces, for BOTH prefill and decode, asserting every
field matches each layer: demand, hit_ops, misses, fetch_ops (order preserved),
fetch_per_rank, routing_map, shadow_after.

Run:  pytest tests/ref/test_ours_vs_ref.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reference import demand_matrix                      # noqa: E402
from ours_reference import OursRefController             # noqa: E402
import prod_driver as pd                                  # noqa: E402

L = 0
FIELDS = ("demand", "hit_ops", "misses", "fetch_ops",
          "fetch_per_rank", "routing_map", "shadow_after")


# ---- deterministic adversarial traces (name, ep, cap, ne, [step dicts]) ----
def _T_basic():
    return ("basic", 2, 3, 8,
            [{0: {0: 5, 1: 4, 2: 3}, 1: {0: 1, 3: 6}},
             {0: {0: 9, 2: 1}, 1: {4: 7, 5: 2}},
             {0: {6: 4, 7: 3}, 1: {0: 2, 6: 8}}])


def _T_skew():
    return ("skew", 4, 4, 16,
            [{0: {1: 9, 2: 1}, 1: {1: 8, 3: 2}, 2: {1: 7}, 3: {1: 6, 4: 1}},
             {0: {1: 2, 5: 9}, 1: {6: 8}, 2: {7: 7, 8: 1}, 3: {9: 5}},
             {0: {1: 3, 5: 4}, 1: {2: 9, 6: 2}, 2: {3: 8}, 3: {4: 7}}])


def _T_single():
    return ("single", 1, 4, 8,
            [{0: {0: 5, 1: 4, 2: 3, 3: 2}},
             {0: {0: 9, 4: 6, 5: 5}},
             {0: {6: 4, 7: 3, 0: 2}}])


def _T_capacity():
    steps = []
    for s in range(6):
        b = (s * 3) % 12
        steps.append({0: {b: 7, (b + 1) % 12: 5, (b + 5) % 12: 2},
                      1: {(b + 2) % 12: 6, b: 1}})
    return ("capacity", 2, 3, 12, steps)


DET_TRACES = [_T_basic(), _T_skew(), _T_single(), _T_capacity()]


def _gen_fuzz(seed):
    rng = np.random.default_rng(seed)
    ep = int(rng.integers(1, 5))
    cap = int(rng.integers(2, 7))
    ne = int(rng.integers(8, 40))
    nsteps = int(rng.integers(3, 7))
    steps = []
    for _ in range(nsteps):
        step = {}
        for r in range(ep):
            ndem = int(rng.integers(0, min(ne, 8)))
            experts = rng.choice(ne, size=ndem, replace=False) if ndem else []
            step[r] = {int(e): int(rng.integers(1, 10)) for e in experts}
        steps.append(step)
    return (f"fuzz{seed}", ep, cap, ne, steps)


FUZZ_TRACES = [_gen_fuzz(s) for s in range(40)]
ALL_TRACES = DET_TRACES + FUZZ_TRACES


def _ids(seq):
    return [t[0] for t in seq]


@pytest.mark.parametrize("is_decode", [False, True], ids=["prefill", "decode"])
@pytest.mark.parametrize("trace", ALL_TRACES, ids=_ids(ALL_TRACES))
def test_ours_prod_matches_oracle(is_decode, trace):
    name, ep, cap, ne, steps = trace
    prod = pd.make_prod_ours(ep, cap, ne)
    ref = OursRefController(ep, cap, ne)

    for si, step in enumerate(steps):
        matrix = demand_matrix(ep, ne, step)
        a = pd.prod_step(prod, matrix, L, is_decode=is_decode)
        b = ref.step(matrix, L, is_decode=is_decode)
        for f in FIELDS:
            assert a[f] == b[f], (
                f"[{name} decode={is_decode} step={si}] field '{f}':\n"
                f"  prod={a[f]}\n  ref ={b[f]}")
        pd.prod_bump(prod)
        ref.bump()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
