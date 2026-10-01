"""EAMC eviction — controller-side priority victim selection (migration).

Verifies the NEW controller-side EAMC eviction (the gap closed this round):
  * PriorityEviction picks the MIN-priority resident expert as victim.
  * this-layer-demanded experts are protected (원본 protected_ondemand).
  * priority=None falls back to LRU (aggregator-off / stride-skip safety).
  * the injected priority actually reaches victim selection in PRODUCTION
    (both via plan_misses(priority=...) and controller.set_layer_priority).
  * LRU vs LFU vs EAMC pick DIFFERENT victims on a crafted trace.
  * production == independent oracle on random priority+demand traces.

Every hand-trace's expected victim is computed in the comments — independent of
both the production code and the (now superseded) golden files.

Run:  pytest tests/ref/test_eamc_eviction.py -q
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


def make_priority(ne, row0: dict, num_layers=64):
    """[num_layers][ne] matrix; row 0 from row0 dict (expert->score), else 1.0.

    Higher score = more important (kept).  Missing experts default 1.0 so an
    unset expert is NOT preferentially evicted unless row0 says so.
    """
    m = [[1.0] * ne for _ in range(num_layers)]
    for e, s in row0.items():
        m[0][e] = float(s)
    return m


# ------------------------------------------------------------------
# 1. min-priority victim.  cap=2, seed [E0,E1].  priority E0=0.9, E1=0.1.
#    demand E2 (miss); E0,E1 not demanded -> none protected.
#    victim = min priority = E1.  shadow [E0, E2].
# ------------------------------------------------------------------
def test_min_priority_victim():
    pr = make_priority(8, {0: 0.9, 1: 0.1})
    c = pd.make_prod(1, 2, 8, "naive", "eamc")
    pd.seed_prod(c, {0: [(L, 0), (L, 1)]})
    a = pd.prod_step(c, demand_matrix(1, 8, {0: {2: 5}}), L, priority=pr)
    assert a["fetch_ops"][0]["victim_expert"] == (L, 1)
    assert a["shadow_after"][0] == [(L, 0), (L, 2)]


# ------------------------------------------------------------------
# 2. protect this-layer demand.  seed [E0,E1].  priority E0=0.1 (lowest!),
#    E1=0.9.  demand E0 (hit) + E2 (miss).  E0 is demanded -> PROTECTED even
#    though it has the lowest priority.  victim = E1.  shadow [E0, E2].
# ------------------------------------------------------------------
def test_protect_this_layer_demand():
    pr = make_priority(8, {0: 0.1, 1: 0.9})
    c = pd.make_prod(1, 2, 8, "naive", "eamc")
    pd.seed_prod(c, {0: [(L, 0), (L, 1)]})
    a = pd.prod_step(c, demand_matrix(1, 8, {0: {0: 5, 2: 3}}), L, priority=pr)
    assert [o["expert"] for o in a["hit_ops"]] == [(L, 0)]
    assert a["fetch_ops"][0]["victim_expert"] == (L, 1)   # E0 protected
    assert a["shadow_after"][0] == [(L, 0), (L, 2)]


# ------------------------------------------------------------------
# 3. priority=None -> LRU fallback.  seed [E0,E1] (E0 older).  demand E2.
#    EAMC with None priority must behave like LRU -> victim E0 (oldest).
# ------------------------------------------------------------------
def test_none_priority_is_lru():
    c = pd.make_prod(1, 2, 8, "naive", "eamc")
    pd.seed_prod(c, {0: [(L, 0), (L, 1)]})
    a = pd.prod_step(c, demand_matrix(1, 8, {0: {2: 5}}), L, priority=None)
    assert a["fetch_ops"][0]["victim_expert"] == (L, 0)   # LRU oldest
    assert a["shadow_after"][0] == [(L, 2), (L, 1)]


# ------------------------------------------------------------------
# 4. set_layer_priority wiring path (live-injection API).
#    Same as #1 but priority injected via controller.set_layer_priority and
#    NOT passed to prod_step -> proves the injected matrix reaches the planner.
# ------------------------------------------------------------------
def test_set_layer_priority_wiring():
    pr = make_priority(8, {0: 0.9, 1: 0.1})
    c = pd.make_prod(1, 2, 8, "naive", "eamc")
    pd.seed_prod(c, {0: [(L, 0), (L, 1)]})
    c.set_layer_priority(pr)                       # inject like ep_executor
    a = pd.prod_step(c, demand_matrix(1, 8, {0: {2: 5}}), L)  # no explicit prio
    assert a["fetch_ops"][0]["victim_expert"] == (L, 1)


# ------------------------------------------------------------------
# 5. LRU vs LFU vs EAMC pick DIFFERENT victims (ACCESS-COUNT LFU).
#    cap=3, seed [E0,E1,E2] (freq 1 each).  Warmup by ACCESS COUNT (token-agnostic):
#      use E0 twice, then E1 once, then E2 once ->
#        freq: E0=3, E1=2, E2=2 ; last_used order: E0(older) < E1 < E2(newest).
#    bump.  Evict step: demand E9 (miss).
#      LRU  -> min last_used = E0 (used least recently)
#      LFU  -> min freq      = E1 (freq2, tie vs E2 broken by last_used E1<E2)
#      EAMC -> min priority  = E2 (priority 0.1 vs 0.5,0.5)
# ------------------------------------------------------------------
def _warm_then_evict(evict, priority=None):
    c = pd.make_prod(1, 3, 16, "naive", evict)
    pd.seed_prod(c, {0: [(L, 0), (L, 1), (L, 2)]})
    for dem in ({0: 1}, {0: 1}, {1: 1}, {2: 1}):   # E0,E0,E1,E2 (counts: 3,2,2)
        pd.prod_step(c, demand_matrix(1, 16, {0: dem}), L, priority=priority)
        pd.prod_bump(c)
    a = pd.prod_step(c, demand_matrix(1, 16, {0: {9: 1}}), L, priority=priority)
    return a["fetch_ops"][0]["victim_expert"]


def test_lru_lfu_eamc_differ():
    pr = make_priority(16, {0: 0.5, 1: 0.5, 2: 0.1})
    assert _warm_then_evict("lru") == (L, 0)
    assert _warm_then_evict("lfu") == (L, 1)
    assert _warm_then_evict("eamc", priority=pr) == (L, 2)


# ------------------------------------------------------------------
# 6. production EAMC == independent oracle, random priority + demand.
# ------------------------------------------------------------------
@pytest.mark.parametrize("ep", [1, 2, 4])
@pytest.mark.parametrize("owner", ["naive", "balanced"])
def test_eamc_prod_matches_oracle(ep, owner):
    N = 150
    for seed in range(N):
        rng = random.Random(seed * 911 + ep * 7 + len(owner))
        cap = rng.choice([1, 2, 3, 4])
        ne = rng.choice([6, 8, 12])
        prod = pd.make_prod(ep, cap, ne, owner, "eamc")
        ref = RefController(ep, cap, ne, owner=owner, evict="eamc")
        for _ in range(rng.randint(2, 6)):
            step = {0: {e: rng.randint(1, 9)
                        for e in rng.sample(range(ne), rng.randint(1, ne))}}
            # random priority each layer (byte-identical to both sides)
            pr = make_priority(ne, {e: round(rng.random(), 3)
                                    for e in range(ne)})
            matrix = demand_matrix(ep, ne, step)
            a = pd.prod_step(prod, matrix, L, priority=pr)
            b = ref.step(matrix, L, priority=pr)
            for f in FIELDS:
                assert a[f] == b[f], (
                    f"[ep={ep} owner={owner} seed={seed} field={f}]\n"
                    f"  prod={a[f]}\n  ref ={b[f]}")
            pd.prod_bump(prod)
            ref.bump()


@pytest.mark.parametrize("ep", [2, 4])
def test_eamc_cross_rank_determinism(ep):
    """Multi-process consistency pillar for EAMC.

    Two independent controllers (= two rank processes) fed the SAME demand and
    the SAME priority matrix must end every layer with BYTE-IDENTICAL shadows
    and identical fetch_ops.  This proves victim selection adds NO hidden
    non-determinism (no dict/set-order dependence), so given an identical
    (all_reduced) priority matrix every rank stays in lockstep -> drift=0.

    The ONE remaining risk is upstream of this code: the priority matrix itself
    being byte-identical across ranks (NCCL float all_reduce).  That must be
    checked on the real multi-rank run; verify_layer_end (drift FATAL) is the
    live safety net that would catch any divergence immediately.
    """
    cap, ne = 3, 12
    rng = random.Random(20260603 + ep)
    c0 = pd.make_prod(ep, cap, ne, "balanced", "eamc")
    c1 = pd.make_prod(ep, cap, ne, "balanced", "eamc")  # the "other rank"
    for _ in range(12):
        step = {0: {e: rng.randint(1, 9)
                    for e in rng.sample(range(ne), rng.randint(1, ne))}}
        pr = make_priority(ne, {e: round(rng.random(), 3) for e in range(ne)})
        matrix = demand_matrix(ep, ne, step)
        a0 = pd.prod_step(c0, matrix, L, priority=pr)
        a1 = pd.prod_step(c1, matrix, L, priority=pr)
        assert a0["shadow_after"] == a1["shadow_after"], "cross-rank shadow drift"
        assert a0["fetch_ops"] == a1["fetch_ops"], "cross-rank plan drift"
        pd.prod_bump(c0)
        pd.prod_bump(c1)


def test_eamc_near_tie_priority_is_deterministic():
    """Near-equal float priorities must NOT make victim selection order-flaky:
    given the SAME matrix, the same victim is chosen every time (tie broken by
    last_used/inserted_at/slot, never by float jitter)."""
    pr = make_priority(16, {0: 0.5000001, 1: 0.5000002, 2: 0.5000000})
    victims = set()
    for _ in range(20):
        c = pd.make_prod(1, 3, 16, "naive", "eamc")
        pd.seed_prod(c, {0: [(L, 0), (L, 1), (L, 2)]})
        a = pd.prod_step(c, demand_matrix(1, 16, {0: {9: 1}}), L, priority=pr)
        victims.add(a["fetch_ops"][0]["victim_expert"])
    assert victims == {(L, 2)}, f"non-deterministic victim under near-tie: {victims}"


if __name__ == "__main__":
    test_min_priority_victim()
    test_protect_this_layer_demand()
    test_none_priority_is_lru()
    test_set_layer_priority_wiring()
    test_lru_lfu_eamc_differ()
    for ep in (1, 2, 4):
        for owner in ("naive", "balanced"):
            test_eamc_prod_matches_oracle(ep, owner)
    print("EAMC eviction: ALL PASS")
