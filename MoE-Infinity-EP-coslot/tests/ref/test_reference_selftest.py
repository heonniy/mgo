"""Oracle self-test — anchors reference.py to HAND-COMPUTED answers.

Every expected value here is worked out by hand in the comments, independent of
both the production controller and the suspect golden files.  If these pass, the
oracle is trustworthy enough to be the differential ground truth.

Run:  pytest tests/ref/test_reference_selftest.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from reference import (  # noqa: E402
    RefController, RefArcher, RefSlotCache, lru_victim, lfu_victim,
    naive_owner, balanced_owner, demand_matrix,
)

L = 0  # expert_layer_id fixed to 0


# ------------------------------------------------------------------
# 1. Empty-cache fill, no eviction.  cap=4, ep=1.
#    demand E5=10, E2=7, E9=3  ->  all miss, processed by (-demand): E5,E2,E9.
#    first_empty_slot picks 0,1,2 in turn.
#    expected fetch_ops: E5->slot0, E2->slot1, E9->slot2 (victim None).
#    shadow_after rank0 = [E5, E2, E9, None].
# ------------------------------------------------------------------
def test_empty_fill_no_evict():
    c = RefController(ep_size=1, cap_per_rank=4, num_experts=16,
                      owner="naive", evict="lru")
    m = demand_matrix(1, 16, {0: {5: 10, 2: 7, 9: 3}})
    r = c.step(m, L)
    assert [op["expert"] for op in r["fetch_ops"]] == [(L, 5), (L, 2), (L, 9)]
    assert [op["dst_slot"] for op in r["fetch_ops"]] == [0, 1, 2]
    assert all(op["victim_expert"] is None for op in r["fetch_ops"])
    assert r["shadow_after"][0] == [(L, 5), (L, 2), (L, 9), None]
    assert r["hit_ops"] == []


# ------------------------------------------------------------------
# 2. LRU victim with hit-refresh.  cap=2, ep=1, seed [E0, E1].
#    After seed: E0.last_used=1, E1.last_used=2 (E1 more recent).
#    Step demand E0=5, E2=3:
#      E0 resident -> hit -> touch (E0.last_used=3, now newest).
#      E2 miss -> cache full -> LRU victim = min last_used = E1 (slot1).
#    expected: victim E1, shadow [E0, E2].
# ------------------------------------------------------------------
def test_lru_hit_refresh_victim():
    c = RefController(1, 2, 16, owner="naive", evict="lru")
    c.seed({0: [(L, 0), (L, 1)]})
    r = c.step(demand_matrix(1, 16, {0: {0: 5, 2: 3}}), L)
    assert len(r["fetch_ops"]) == 1
    op = r["fetch_ops"][0]
    assert op["expert"] == (L, 2)
    assert op["victim_expert"] == (L, 1)   # E1 evicted, E0 refreshed by hit
    assert op["dst_slot"] == 1
    assert r["shadow_after"][0] == [(L, 0), (L, 2)]


# ------------------------------------------------------------------
# 3. LRU vs LFU pick DIFFERENT victims.  cap=2, ep=1, seed [E0, E1].
#    ACCESS-COUNT LFU: frequency = # of demand events (token-agnostic), so we
#    differentiate by HOW MANY TIMES each expert is used, not by token volume.
#    Use E0 twice (count high) then E1 once last (count low, most recent):
#       E0: freq=3 (seed+2), last_used older ; E1: freq=2 (seed+1), last_used newest
#    Step (evict E2):
#       LRU: min last_used -> E0 (used least recently) -> victim E0
#       LFU: min freq      -> E1 (2 < 3, fewer accesses) -> victim E1
# ------------------------------------------------------------------
def _two_step_then_evict(evict):
    c = RefController(1, 2, 16, owner="naive", evict=evict)
    c.seed({0: [(L, 0), (L, 1)]})
    c.step(demand_matrix(1, 16, {0: {0: 1}}), L); c.bump()  # E0 used
    c.step(demand_matrix(1, 16, {0: {0: 1}}), L); c.bump()  # E0 used again (count↑)
    c.step(demand_matrix(1, 16, {0: {1: 1}}), L); c.bump()  # E1 used last (recent)
    r = c.step(demand_matrix(1, 16, {0: {2: 1}}), L)        # evict one of {E0,E1}
    return r["fetch_ops"][0]["victim_expert"]


def test_lru_lfu_differ():
    assert _two_step_then_evict("lru") == (L, 0)   # LRU evicts least-recent E0
    assert _two_step_then_evict("lfu") == (L, 1)   # LFU evicts fewer-access E1


# ------------------------------------------------------------------
# 4. naive owner = expert_id % ep_size.  ep=2, miss {E0,E1,E2,E3}.
#    rank0 = {E0,E2}, rank1 = {E1,E3}.
# ------------------------------------------------------------------
def test_naive_owner_rule():
    miss = [(L, 0), (L, 1), (L, 2), (L, 3)]
    out = naive_owner(miss, 2)
    assert set(out[0]) == {(L, 0), (L, 2)}
    assert set(out[1]) == {(L, 1), (L, 3)}


# ------------------------------------------------------------------
# 5. balanced owner splits evenly, max-min <= 1.  ep=2, 3 misses.
# ------------------------------------------------------------------
def test_balanced_owner_even():
    # balanced = count-equal(±1) seeded-random round-robin (2026-06-10 정의).
    # membership 은 seed 의존 random → 성질만 검증: 개수 ±1, 보존, 결정성.
    miss = [(L, 0), (L, 1), (L, 2)]
    out = balanced_owner(miss, 2)
    counts = sorted(len(v) for v in out.values())
    assert counts == [1, 2]
    assert max(counts) - min(counts) == 1
    got = sorted(k for v in out.values() for k in v)
    assert got == sorted(miss)                 # 보존 (중복/누락 없음)
    assert balanced_owner(miss, 2) == out      # 결정성 (동일 입력 → 동일 출력)


# ------------------------------------------------------------------
# 6. naive can pile onto one rank where balanced would not.
#    ep=4, miss experts all == 0 mod 4: {E0,E4,E8,E12}.
#    naive: ALL go to rank0 (imbalance 4-0).
#    balanced: one per rank (imbalance 0).
# ------------------------------------------------------------------
def test_naive_pileup_balanced_spreads():
    miss = [(L, 0), (L, 4), (L, 8), (L, 12)]
    nv = naive_owner(miss, 4)
    nv_counts = [len(nv[r]) for r in range(4)]
    assert nv_counts == [4, 0, 0, 0]            # naive piles on rank0

    bl = balanced_owner(miss, 4)
    bl_counts = [len(bl[r]) for r in range(4)]
    assert bl_counts == [1, 1, 1, 1]            # balanced spreads evenly
    assert max(bl_counts) - min(bl_counts) == 0


# ------------------------------------------------------------------
# 7. RefArcher: direct vs staging + controller-victim-only.
#    cap=4, seed [E0,E1,E2,E3].
#    plan: hit E0@slot0; miss E9 -> slot1 victim E1.
#    slot1 is NOT being computed (E1 not a hit) -> DIRECT.
# ------------------------------------------------------------------
def test_archer_direct_when_victim_idle():
    a = RefArcher(4)
    a.seed_slots([(L, 0), (L, 1), (L, 2), (L, 3)])
    a.submit_plan(hit_tuples=[(L, 0, 0)],
                  miss_tuples=[(L, 9, 1, L, 1, 0)])
    assert a.direct == 1 and a.staging == 0
    assert a.get_cached_slots() == [(L, 0), (L, 9), (L, 2), (L, 3)]


# ------------------------------------------------------------------
# 8. RefArcher: staging when victim slot is being computed (hit on it).
#    cap=4, seed [E0,E1,E2,E3].
#    plan: hit E1@slot1 (slot1 COMPUTING); miss E9 -> slot1 victim E1.
#    dst_slot busy -> STAGING; commit only after wait_victim_compute_done.
# ------------------------------------------------------------------
def test_archer_staging_when_victim_busy():
    a = RefArcher(4)
    a.seed_slots([(L, 0), (L, 1), (L, 2), (L, 3)])
    a.submit_plan(hit_tuples=[(L, 1, 1)],
                  miss_tuples=[(L, 9, 1, L, 1, 0)])
    assert a.staging == 1 and a.direct == 0
    names = [e["event"] for e in a.events]
    # commit must come AFTER wait_victim_compute_done
    assert names.index("wait_victim_compute_done") < names.index("commit")
    assert a.get_cached_slots() == [(L, 0), (L, 9), (L, 2), (L, 3)]


# ------------------------------------------------------------------
# 9. RefArcher refuses to invent a victim (controller under-planned).
#    seed slot1 = E1; miss E9 -> slot1 with NO victim (vl=ve=-1).
#    Must raise EMPTY-SLOT MISMATCH.
# ------------------------------------------------------------------
def test_archer_refuses_self_victim():
    a = RefArcher(4)
    a.seed_slots([(L, 0), (L, 1), None, None])
    raised = False
    try:
        a.submit_plan(hit_tuples=[], miss_tuples=[(L, 9, 1, -1, -1, 0)])
    except AssertionError as ex:
        raised = "EMPTY-SLOT MISMATCH" in str(ex)
    assert raised, "archer must not overwrite an occupied slot with no victim"


# ------------------------------------------------------------------
# 10. RefArcher catches victim mismatch (plan victim != resident).
# ------------------------------------------------------------------
def test_archer_victim_mismatch():
    a = RefArcher(4)
    a.seed_slots([(L, 0), (L, 1), (L, 2), (L, 3)])
    raised = False
    try:
        # claim victim E7 at slot1 but slot1 holds E1
        a.submit_plan(hit_tuples=[], miss_tuples=[(L, 9, 1, L, 7, 0)])
    except AssertionError as ex:
        raised = "VICTIM MISMATCH" in str(ex)
    assert raised


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("oracle self-test: ALL PASS")
