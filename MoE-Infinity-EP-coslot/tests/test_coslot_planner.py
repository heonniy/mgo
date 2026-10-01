"""Golden Test A — Controller planner golden (revision.md §1.5).

Pure-Python (no model / no archer).  Verifies that RankPlanner produces the
exact FetchPlan revision.md specifies for the tight cap=4 scenario, with:
  * correct dst_slot / victim / rank-local order per op,
  * correct cache_view state after EACH op (re-simulated),
  * LRU victim chain (O0,O1 before the just-touched hits H0,H1),
  * no duplicate-victim / no immediate re-eviction of a just-inserted expert,
  * a tight stress scenario (cap=4, hit=2, miss≫cap) exercising the victim chain.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_infinity_ep.controller.evict_policy import LRUEviction
from moe_infinity_ep.controller.rank_planner import RankPlanner
from moe_infinity_ep.controller.slot_cache import RankSlotCache


L = 0  # single layer for this test
def E(e):
    return (L, e)


def _build_initial_cache():
    """cap=4, slot0 H0, slot1 H1, slot2 O0, slot3 O1, hits H0/H1 touched."""
    rc = RankSlotCache(rank=0, cap=4)
    # apply in slot order so inserted_at/last_used are deterministic.
    rc.apply(slot=0, evict=None, insert=E(100), demand_count=0)  # H0
    rc.apply(slot=1, evict=None, insert=E(101), demand_count=0)  # H1
    rc.apply(slot=2, evict=None, insert=E(200), demand_count=0)  # O0
    rc.apply(slot=3, evict=None, insert=E(201), demand_count=0)  # O1
    # classify touches the current-layer hits (H0 then H1) → most-recently-used.
    rc.touch_use(E(100), demand_count=10)
    rc.touch_use(E(101), demand_count=10)
    return rc


def test_planner_golden():
    rc = _build_initial_cache()
    planner = RankPlanner(evict_policy=LRUEviction())

    # miss experts; demand strictly decreasing so the sort yields exactly
    # M3, M0, M6, M1, M2, M7 (revision.md §1.5).
    H0, H1, O0, O1 = E(100), E(101), E(200), E(201)
    M3, M0, M6, M1, M2, M7 = E(3), E(0), E(6), E(1), E(2), E(7)
    demand = {M3: 60, M0: 50, M6: 40, M1: 30, M2: 20, M7: 10}

    fetch_ops = planner.plan(
        rank=0, miss_experts=list(demand.keys()), rank_cache=rc, demand=demand)

    # ---- golden op list: (expert, dst_slot, victim, order) ----
    golden = [
        (M3, 2, O0, 0),
        (M0, 3, O1, 1),
        (M6, 0, H0, 2),
        (M1, 1, H1, 3),
        (M2, 2, M3, 4),
        (M7, 3, M0, 5),
    ]
    assert len(fetch_ops) == len(golden), (
        f"expected {len(golden)} ops, got {len(fetch_ops)}")
    for op, (exp_e, exp_slot, exp_victim, exp_order) in zip(fetch_ops, golden):
        assert op.expert == exp_e, f"expert {op.expert} != {exp_e}"
        assert op.dst_slot == exp_slot, (
            f"{op.expert}: dst_slot {op.dst_slot} != {exp_slot}")
        assert op.victim_expert == exp_victim, (
            f"{op.expert}: victim {op.victim_expert} != {exp_victim}")
        assert op.order == exp_order, (
            f"{op.expert}: order {op.order} != {exp_order}")

    # ---- re-simulate per-op cache state from a fresh cache, asserting the
    # victim actually occupies dst_slot right before each op (golden state). ----
    sim = _build_initial_cache()
    seen_victims = set()
    for op in fetch_ops:
        # victim must currently occupy dst_slot.
        assert sim.slots[op.dst_slot] == op.victim_expert, (
            f"op {op.order}: slot {op.dst_slot} holds {sim.slots[op.dst_slot]}, "
            f"victim says {op.victim_expert}")
        # no duplicate victim across ops (each evicted at most once).
        if op.victim_expert is not None:
            assert op.victim_expert not in seen_victims, (
                f"duplicate victim {op.victim_expert}")
            seen_victims.add(op.victim_expert)
        # a just-inserted expert must not be the immediate next victim.
        sim.apply(slot=op.dst_slot, evict=op.victim_expert, insert=op.expert,
                  demand_count=demand.get(op.expert, 0))

    # ---- final cache_view: {M6@0, M1@1, M2@2, M7@3} ----
    final = {sim.slots[i] for i in range(4)}
    assert final == {M6, M1, M2, M7}, f"final cache {final}"
    assert sim.slots[0] == M6 and sim.slots[1] == M1
    assert sim.slots[2] == M2 and sim.slots[3] == M7


def test_planner_empty_slots_first():
    """Empty slots are filled (lowest id first) before any eviction."""
    rc = RankSlotCache(rank=0, cap=4)
    rc.apply(slot=0, evict=None, insert=E(100), demand_count=0)  # H0 only
    planner = RankPlanner(evict_policy=LRUEviction())
    miss = [E(5), E(6), E(7)]
    demand = {E(5): 30, E(6): 20, E(7): 10}
    ops = planner.plan(rank=0, miss_experts=miss, rank_cache=rc, demand=demand)
    # 3 empty slots (1,2,3) filled in order, no victims.
    assert [(o.expert, o.dst_slot, o.victim_expert, o.order) for o in ops] == [
        (E(5), 1, None, 0), (E(6), 2, None, 1), (E(7), 3, None, 2)]


def test_planner_already_resident_skip():
    """A demanded expert already resident is skipped (defensive)."""
    rc = RankSlotCache(rank=0, cap=4)
    rc.apply(slot=0, evict=None, insert=E(5), demand_count=0)
    planner = RankPlanner(evict_policy=LRUEviction())
    ops = planner.plan(rank=0, miss_experts=[E(5), E(9)],
                       rank_cache=rc, demand={E(5): 99, E(9): 1})
    # E(5) skipped (resident); only E(9) fetched into an empty slot.
    assert len(ops) == 1 and ops[0].expert == E(9) and ops[0].order == 0


if __name__ == "__main__":
    test_planner_golden()
    test_planner_empty_slots_first()
    test_planner_already_resident_skip()
    print("Test A (planner golden): ALL PASS")
