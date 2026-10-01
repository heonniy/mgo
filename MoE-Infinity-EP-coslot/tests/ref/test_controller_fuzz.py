"""Phase 2 — inline differential fuzz: production controller vs oracle.

Generates many deterministic (seeded) random traces and asserts the production
controller matches the oracle on every field at every layer.  No stored golden;
the oracle is computed inline.  On mismatch the seed + minimal context is
printed so the failure reproduces exactly.

Run:  pytest tests/ref/test_controller_fuzz.py -q
      MOE_FUZZ_SEEDS=2000 pytest tests/ref/test_controller_fuzz.py -q   # deeper
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reference import RefController, demand_matrix          # noqa: E402
import prod_driver as pd                                     # noqa: E402

L = 0
FIELDS = ("demand", "hit_ops", "miss_per_rank", "misses",
          "fetch_ops", "routing_map", "shadow_after")

N_SEEDS = int(os.environ.get("MOE_FUZZ_SEEDS", "400"))
OWNERS = ["naive", "balanced"]
EVICTS = ["lru", "lfu"]


def _gen_trace(rng):
    ep = rng.choice([1, 2, 4])
    cap = rng.choice([1, 2, 3, 4])
    ne = rng.choice([6, 8, 12, 16])
    n_steps = rng.randint(2, 8)
    steps = []
    for _ in range(n_steps):
        # pick a random set of demanded experts and counts, spread over ranks
        k = rng.randint(1, ne)
        experts = rng.sample(range(ne), k)
        step = {r: {} for r in range(ep)}
        for e in experts:
            r = rng.randrange(ep)
            step[r][e] = rng.randint(1, 20)
        steps.append(step)
    return ep, cap, ne, steps


def _run_one(seed, owner, evict):
    rng = random.Random(seed * 131 + hash((owner, evict)) % 997)
    ep, cap, ne, steps = _gen_trace(rng)
    prod = pd.make_prod(ep, cap, ne, owner, evict)
    ref = RefController(ep, cap, ne, owner=owner, evict=evict)
    for si, step in enumerate(steps):
        matrix = demand_matrix(ep, ne, step)
        a = pd.prod_step(prod, matrix, L)
        b = ref.step(matrix, L)
        for f in FIELDS:
            if a[f] != b[f]:
                return (f"seed={seed} owner={owner} evict={evict} ep={ep} "
                        f"cap={cap} ne={ne} step={si} field={f}\n"
                        f"  prod={a[f]}\n  ref ={b[f]}\n  steps={steps}")
        # cheap invariants
        if len(a["demand"]) != len(a["hit_ops"]) + len(a["misses"]):
            return f"seed={seed} accounting gud!=hit+miss steps={steps}"
        if len(a["misses"]) != len(a["fetch_ops"]):
            return f"seed={seed} miss!=fetch steps={steps}"
        pd.prod_bump(prod)
        ref.bump()
    return None


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("evict", EVICTS)
def test_fuzz(owner, evict):
    fails = []
    for seed in range(N_SEEDS):
        err = _run_one(seed, owner, evict)
        if err is not None:
            fails.append(err)
            if len(fails) >= 3:
                break
    assert not fails, (
        f"{len(fails)} fuzz mismatch(es) (owner={owner} evict={evict}):\n"
        + "\n---\n".join(fails))


if __name__ == "__main__":
    total = 0
    for o in OWNERS:
        for e in EVICTS:
            for s in range(N_SEEDS):
                err = _run_one(s, o, e)
                assert err is None, err
                total += 1
    print(f"fuzz: {total} traces ALL PASS")
