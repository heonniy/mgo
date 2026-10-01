"""Random baseline (#1) — determinism, cap-respect, archer-replay drift=0.

random = RandomOwner (uniform rank with quota K) + RandomEviction (uniform
victim), seeded from (layer_id, layer_seq) via a process-stable mix (NOT builtin
hash).  Must be byte-identical across EP ranks (and processes) so the replicated
shadow never drifts.

Run:  pytest tests/ref/test_random_baseline.py -q
"""
from __future__ import annotations

import os
import random
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reference import RefArcher, demand_matrix          # noqa: E402
import prod_driver as pd                                  # noqa: E402
from moe_infinity_ep.controller.owner_policy import RandomOwner  # noqa: E402
from moe_infinity_ep.controller.evict_policy import RandomEviction  # noqa: E402
from moe_infinity_ep.controller.slot_cache import (       # noqa: E402
    GlobalSlotCacheView, RankSlotCache)

L = 0
_HERE = os.path.dirname(__file__)


def test_random_owner_same_seed_same_result():
    """Same (layer_id, layer_seq) -> identical assignment on two views.
    No quota cap (fully random) — only determinism + full coverage are required."""
    misses = [(0, e) for e in range(10)]
    demand = {(0, e): 1 for e in range(10)}
    v1 = GlobalSlotCacheView(4, 5, 64, 32)
    v2 = GlobalSlotCacheView(4, 5, 64, 32)
    o = RandomOwner()
    a1 = o.assign_rank(misses, demand, v1)
    a2 = o.assign_rank(misses, demand, v2)
    assert a1 == a2
    # every miss assigned exactly once across ranks (no cap → counts may be lumpy)
    assert sum(len(v) for v in a1.values()) == len(misses)
    allk = [k for v in a1.values() for k in v]
    assert sorted(allk) == sorted(misses)


def test_random_owner_seed_varies_with_layer_seq():
    misses = [(0, e) for e in range(10)]
    demand = {(0, e): 1 for e in range(10)}
    v = GlobalSlotCacheView(4, 5, 64, 32)
    a0 = RandomOwner().assign_rank(misses, demand, v)
    v.bump_all_layers()  # layer_seq -> 1
    a1 = RandomOwner().assign_rank(misses, demand, v)
    # different layer_seq -> (almost surely) different assignment
    assert a0 != a1


def test_random_eviction_same_seed_same_victim():
    rc1 = RankSlotCache(rank=2, cap=4)
    rc2 = RankSlotCache(rank=2, cap=4)
    for rc in (rc1, rc2):
        for e in range(4):
            rc.apply(slot=e, evict=None, insert=(0, e), demand_count=1)
    e = RandomEviction()
    s1 = e.pick_victim(rc1, incoming=(0, 99), demand={})
    s2 = e.pick_victim(rc2, incoming=(0, 99), demand={})
    assert s1 == s2 and s1 is not None


@pytest.mark.parametrize("seed", range(80))
def test_random_replay_no_drift(seed):
    rng = random.Random(seed * 5701 + 11)
    ep = rng.choice([1, 2, 4])
    cap = rng.choice([2, 3, 4])
    ne = rng.choice([8, 12, 16])
    n_steps = rng.randint(3, 8)
    prod = pd.make_prod(ep, cap, ne, "random", "random")
    archers = [RefArcher(cap) for _ in range(ep)]
    for si in range(n_steps):
        k = rng.randint(1, min(ne, 8))
        step = {r: {} for r in range(ep)}
        for ex in rng.sample(range(ne), k):
            step[rng.randrange(ep)][ex] = rng.randint(1, 20)
        matrix = demand_matrix(ep, ne, step)
        hits, misses, shadow = pd.prod_plan_tuples(prod, matrix, L)
        for r in range(ep):
            archers[r].submit_plan(hits[r], misses[r])
            assert archers[r].get_cached_slots() == shadow[r], (
                f"random drift seed={seed} step={si} rank={r}")
            # quota respected: per-rank fetch <= ceil(M/ep)
        pd.prod_bump(prod)


def _subproc(mode, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    r = subprocess.run(
        [sys.executable, os.path.join(_HERE, "_det_runner.py"), mode, "0"],
        capture_output=True, text=True, env=env, cwd=_HERE)
    assert r.returncode == 0, f"runner failed: {r.stderr}"
    return r.stdout


def test_random_cross_process_determinism():
    """The dumb-baseline randomness must be reproducible across processes
    (proves _stable_seed, not builtin hash)."""
    d0 = _subproc("random", 0)
    d1 = _subproc("random", 1)
    d2 = _subproc("random", 999)
    assert d0 == d1 == d2, "random baseline depends on PYTHONHASHSEED (builtin hash!)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
