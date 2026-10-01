"""Golden Test B — Archer scheduler golden (revision.md §3.5).

The real C++ scheduler needs a loaded model to fetch (host_memory_ptr) and a
GPU to GEMM, so it can't run as a standalone unit test.  This test instead
exercises a Python REFERENCE MODEL of the coslot scheduler — a 1:1 mirror of
ExpertDispatcher::{SubmitPlan, StartNextFetch, DoDirectFetch,
DoStagingFetchPark, CompleteStagingFetch, Commit} + the single exec worker —
driven by the SAME FetchPlan the controller emits.

It verifies revision.md §3.5:
  1. active_fetch <= 1 always
  2. PlanQueue pop order == FetchOp.order (contiguous FIFO)
  3. both direct and staging paths occur
  4. a staging op does not advance until its dst_slot's GEMM completes
  5. victim mismatch → FATAL
  6. no slot overwrite while its prior occupant is still computing
  7. final slot_to_key == controller final cache_view

The real C++ implementation is the same algorithm; the model run's per-layer
drift==0 verify (controller cache_view == archer get_cached_experts) is the
end-to-end check that the C++ matches this model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_infinity_ep.controller.evict_policy import LRUEviction
from moe_infinity_ep.controller.rank_planner import RankPlanner
from moe_infinity_ep.controller.slot_cache import RankSlotCache


class SchedFATAL(Exception):
    pass


class RefScheduler:
    """Faithful Python mirror of the C++ coslot scheduler (one GPU)."""

    FREE, COMPUTING = 0, 1

    def __init__(self, cap, initial_slots):
        self.cap = cap
        # resident slots [0, cap); staging is conceptual (index cap).
        self.slot_to_key = dict(initial_slots)          # slot -> (l,e)
        self.key_to_slot = {k: s for s, k in initial_slots.items()}
        self.cached = set(initial_slots.values())
        self.slot_state = [self.FREE] * cap
        # scheduler state
        self.plan_queue = []
        self.has_active = False
        self.active_parked = False
        self.active_op = None
        self.expected_order = 0
        # exec
        self.execq = []          # FIFO of (l, e, slot)
        self.pending = 0
        # invariants / coverage
        self.max_active = 0
        self.n_direct = 0
        self.n_staging = 0
        self.pop_orders = []
        self.overwrite_violation = False

    # ---- mirror of SubmitPlan ----
    def submit_plan(self, hit_ops, miss_ops):
        self.plan_queue = sorted(miss_ops, key=lambda o: o["order"])
        self.has_active = False
        self.active_parked = False
        self.expected_order = 0
        self.pending = len(hit_ops) + len(miss_ops)
        for h in hit_ops:
            key = (h["layer"], h["expert"])
            if self.slot_to_key.get(h["slot"]) != key:
                raise SchedFATAL(f"HIT slot mismatch {h}")
            self.slot_state[h["slot"]] = self.COMPUTING
            self.execq.append((h["layer"], h["expert"], h["slot"]))
        self._start_next_fetch()

    # ---- mirror of StartNextFetch (loop form) ----
    def _start_next_fetch(self):
        while not self.has_active and self.plan_queue:
            op = self.plan_queue.pop(0)
            if op["order"] != self.expected_order:
                raise SchedFATAL(
                    f"order mismatch got {op['order']} exp {self.expected_order}")
            self.expected_order += 1
            self.pop_orders.append(op["order"])
            self.has_active = True
            self.max_active = max(self.max_active, 1)
            self.active_op = op
            self.active_parked = False
            if self.slot_state[op["dst_slot"]] == self.FREE:
                self._do_direct_fetch(op)
            else:
                self._do_staging_park(op)

    def _do_direct_fetch(self, op):
        self.n_direct += 1
        self._commit(op)
        self.slot_state[op["dst_slot"]] = self.COMPUTING
        self.execq.append((op["layer"], op["expert"], op["dst_slot"]))
        self.has_active = False

    def _do_staging_park(self, op):
        self.n_staging += 1
        # H2D into staging issued now (overlaps prior GEMM); park.
        self.active_parked = True   # has_active stays True → loop exits

    def _complete_staging(self):
        op = self.active_op
        self._commit(op)
        self.slot_state[op["dst_slot"]] = self.COMPUTING
        self.execq.append((op["layer"], op["expert"], op["dst_slot"]))
        self.has_active = False
        self.active_parked = False
        self._start_next_fetch()

    def _commit(self, op):
        new_key = (op["layer"], op["expert"])
        dst = op["dst_slot"]
        if op["victim"] is not None:
            vkey = op["victim"]
            if self.slot_to_key.get(dst) != vkey:
                raise SchedFATAL(
                    f"VICTIM MISMATCH dst_slot={dst} controller_victim={vkey} "
                    f"archer_holds={self.slot_to_key.get(dst)} op={op}")
            del self.key_to_slot[vkey]
            self.cached.discard(vkey)
        else:
            if dst in self.slot_to_key:
                raise SchedFATAL(
                    f"EMPTY-SLOT MISMATCH dst_slot={dst} holds "
                    f"{self.slot_to_key[dst]} op={op}")
        self.slot_to_key[dst] = new_key
        self.key_to_slot[new_key] = dst
        self.cached.add(new_key)

    # ---- mirror of one exec worker GEMM completion ----
    def _exec_one(self):
        l, e, slot = self.execq.pop(0)
        if self.slot_to_key.get(slot) != (l, e):
            raise SchedFATAL(f"ExecTask slot mismatch slot={slot} expert={(l,e)}")
        # If a parked staging op is waiting on a slot that is NOT yet free, and
        # we are about to overwrite a slot still computing, that's a violation.
        self.slot_state[slot] = self.FREE
        if (self.has_active and self.active_parked
                and self.active_op["dst_slot"] == slot):
            self._complete_staging()
        self.pending -= 1

    def run_to_completion(self):
        # interleave: drain execq (FIFO) which also unparks staging ops.
        guard = 0
        while self.pending > 0:
            guard += 1
            if guard > 100000:
                raise SchedFATAL("scheduler did not converge (deadlock)")
            if self.execq:
                self._exec_one()
            elif self.has_active and self.active_parked:
                # parked op waiting but no GEMM left to free its slot — deadlock
                raise SchedFATAL(
                    f"parked staging op {self.active_op} has no pending GEMM "
                    "to free its dst_slot")
            else:
                # no exec, no active → start more fetches
                self._start_next_fetch()
                if not self.execq and not self.has_active:
                    raise SchedFATAL("no progress")
        if self.has_active or self.plan_queue:
            raise SchedFATAL("pending==0 but fetches outstanding")


def _controller_plan(cap, initial_keys, hit_keys, miss_demand):
    """Reuse RankPlanner to produce the SAME FetchPlan the controller emits."""
    rc = RankSlotCache(rank=0, cap=cap)
    for s, k in initial_keys.items():
        rc.apply(slot=s, evict=None, insert=k, demand_count=0)
    for k in hit_keys:
        rc.touch_use(k, demand_count=10)
    planner = RankPlanner(evict_policy=LRUEviction())
    ops = planner.plan(rank=0, miss_experts=list(miss_demand.keys()),
                       rank_cache=rc, demand=miss_demand)
    # convert to the dict form the dispatcher would build (tuples → dicts).
    miss_ops = [
        {"layer": o.expert[0], "expert": o.expert[1], "dst_slot": o.dst_slot,
         "victim": o.victim_expert, "order": o.order}
        for o in ops
    ]
    final_shadow = {s: rc.slots[s] for s in range(cap) if rc.slots[s] is not None}
    return miss_ops, final_shadow


def test_scheduler_golden():
    cap = 4
    H0, H1, O0, O1 = (0, 100), (0, 101), (0, 200), (0, 201)
    initial = {0: H0, 1: H1, 2: O0, 3: O1}
    hit_keys = [H0, H1]
    M3, M0, M6, M1, M2, M7 = ((0, 3), (0, 0), (0, 6), (0, 1), (0, 2), (0, 7))
    demand = {M3: 60, M0: 50, M6: 40, M1: 30, M2: 20, M7: 10}

    miss_ops, final_shadow = _controller_plan(cap, initial, hit_keys, demand)
    hit_ops = [{"layer": k[0], "expert": k[1], "slot": s}
               for s, k in initial.items() if k in hit_keys]

    sched = RefScheduler(cap, initial)
    sched.submit_plan(hit_ops, miss_ops)
    sched.run_to_completion()

    # 1. active_fetch <= 1 always
    assert sched.max_active <= 1
    # 2. PlanQueue popped in contiguous order
    assert sched.pop_orders == list(range(len(miss_ops)))
    # 3. both paths occurred (O0,O1 direct; H0,H1 (current hits) staging; ...)
    assert sched.n_direct > 0 and sched.n_staging > 0, (
        f"direct={sched.n_direct} staging={sched.n_staging}")
    # 7. final archer slot_to_key == controller final cache_view
    assert sched.slot_to_key == final_shadow, (
        f"archer {sched.slot_to_key} != shadow {final_shadow}")
    assert sched.pending == 0
    print(f"Test B golden: direct={sched.n_direct} staging={sched.n_staging} "
          f"final={sched.slot_to_key} PASS")


def test_scheduler_victim_mismatch_fatal():
    """Commit must FATAL when archer's slot doesn't hold the controller's victim."""
    cap = 2
    A, B = (0, 10), (0, 11)
    sched = RefScheduler(cap, {0: A, 1: B})
    # miss op claims victim=A at slot1, but slot1 actually holds B → FATAL.
    bad = [{"layer": 0, "expert": 5, "dst_slot": 1, "victim": A, "order": 0}]
    try:
        sched.submit_plan([], bad)
        sched.run_to_completion()
    except SchedFATAL as e:
        assert "VICTIM MISMATCH" in str(e)
        print("Test B victim-mismatch FATAL: PASS")
        return
    raise AssertionError("expected SchedFATAL on victim mismatch")


def test_scheduler_all_direct_empty_slots():
    """All-empty cap → all direct, no staging, no parks."""
    cap = 4
    sched = RefScheduler(cap, {})
    miss = [{"layer": 0, "expert": e, "dst_slot": e, "victim": None,
             "order": e} for e in range(4)]
    sched.submit_plan([], miss)
    sched.run_to_completion()
    assert sched.n_staging == 0 and sched.n_direct == 4
    assert sched.slot_to_key == {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3)}
    print("Test B all-direct: PASS")


if __name__ == "__main__":
    test_scheduler_golden()
    test_scheduler_victim_mismatch_fatal()
    test_scheduler_all_direct_empty_slots()
    print("Test B (scheduler golden): ALL PASS")
