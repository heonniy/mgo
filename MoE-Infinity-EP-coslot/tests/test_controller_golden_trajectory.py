"""Golden Test 1 — Controller exact trajectory (moe_cache_golden README §1).

Drives the REAL ``GlobalCacheController`` (no NCCL / no archer) through the
synthetic demand trajectory in ``controller_gold_trajectory.jsonl`` and asserts
the *exact* cache trajectory against the golden answer — not hit rate, not
latency, but every field:

    global_unique_demand -> hit_ops -> misses -> miss_per_rank -> fetch_ops
    -> routing_map -> shadow_after, plus structural invariants.

Terminology (mandatory, per README):
  * trace_step       — synthetic cache trajectory index (golden line index).
  * runtime_layer_id — layer id passed to controller APIs (== trace_step).
  * expert_layer_id  — layer component of ExpertKey(layer_id, expert_id),
                       FIXED to 0 here.  Every key is (0, expert_id).

To honour expert_layer_id == 0 regardless of runtime_layer_id, the controller
is always called with ``layer_id=0`` so every ExpertKey it builds is (0, e).
The trace_step is only the golden line selector.

Two run modes:
  * sequential — one controller, steps 0->1->2->3->4 cumulatively; each step's
    shadow_before must equal the previous step's shadow_after.
  * isolated   — fresh controller per step, shadow_before seeded directly.

NB: the two modes can diverge on LRU victim selection (cross-step recency
history vs a freshly-seeded slot-order recency).  Each mode is asserted
against the golden independently; see the module-level analysis in the
accompanying report.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from moe_infinity_ep.controller.global_controller import GlobalCacheController
from moe_infinity_ep.controller.owner_policy import build_owner_policy
from moe_infinity_ep.controller.evict_policy import build_evict_policy
from moe_infinity_ep.controller.rank_planner import RankPlanner


GOLD_DIR = os.environ.get(
    "MOE_EP_GOLD_DIR", "/home/work/hyewon.lee/실험/moe_cache_golden")
GOLD_FILE = os.path.join(GOLD_DIR, "controller_gold_trajectory.jsonl")

# Superseded 2026-06-03: the golden was moved to ``superseded/`` and this test is
# replaced by the import-pure independent oracle suite at ``tests/ref/``
# (test_controller_vs_ref.py / test_golden_regression.py).  Skip cleanly if the
# old golden is absent instead of erroring at collection.
if not os.path.exists(GOLD_FILE):
    pytest.skip("old golden moved to superseded/ — replaced by tests/ref/ "
                "independent oracle suite", allow_module_level=True)

EXPERT_LAYER_ID = 0          # README: every ExpertKey is (0, expert_id)
NUM_EXPERTS = 32             # max expert id in golden is 23
NUM_LAYERS = 64


def _load_gold():
    with open(GOLD_FILE) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


GOLD = _load_gold()
CFG = GOLD[0]["config"]
EP_SIZE = CFG["ep_size"]
CAP = CFG["cap_per_rank"]


# ============================================================
# SyntheticDemandCollector — README §1: convert per_rank_demand into the
# exact per_rank_count tensor that DemandCollector.collect_with_count returns.
# ============================================================
class SyntheticDemandCollector:
    """Returns a pre-loaded ``[ep_size, num_experts]`` int64 demand matrix.

    Mirrors the real ``DemandCollector`` attributes/method so it drops into
    ``GlobalCacheController.demand_collector``.  Uses expert_layer_id=0 for
    ExpertKey construction implicitly: it only carries expert_id counts; the
    controller stamps layer_id from the (always 0) layer_id argument.
    """

    def __init__(self, ep_size: int, num_experts: int):
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.ep_group = None
        self._current = None

    def load_step(self, per_rank_demand: dict) -> None:
        t = torch.zeros(self.ep_size, self.num_experts, dtype=torch.int64)
        for r_str, demands in per_rank_demand.items():
            r = int(r_str)
            for e_str, count in demands.items():
                t[r, int(e_str)] = int(count)
        self._current = t

    def collect_with_count(self, layer_id, local_router_mask):
        assert self._current is not None, "load_step() not called"
        return self._current


# ============================================================
# normalization — actual controller output -> golden JSON shape
# ============================================================
def _ek(key):
    """ExpertKey (l, e) -> [l, e]; None -> None."""
    return None if key is None else [int(key[0]), int(key[1])]


def _make_controller():
    return GlobalCacheController(
        ep_size=EP_SIZE, ep_rank=0, cap_per_rank=CAP,
        num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS,
        ep_group=None,
        owner_policy=build_owner_policy(CFG["owner_policy"]),
        rank_planner=RankPlanner(build_evict_policy(CFG["evict_policy"])),
        demand_collector=SyntheticDemandCollector(EP_SIZE, NUM_EXPERTS),
        command_dispatcher=None,   # no archer -> verify_layer_end just bumps
    )


def _shadow(controller):
    return {
        str(r): [_ek(s) for s in controller.cache_view.per_rank[r].slots]
        for r in range(EP_SIZE)
    }


def _run_step(controller, line):
    """Run one trajectory step on ``controller``; return the actual outputs in
    golden JSON shape.  Always uses layer_id=0 (expert_layer_id semantics)."""
    controller.demand_collector.load_step(line["per_rank_demand"])
    dummy_mask = torch.zeros(1, NUM_EXPERTS, dtype=torch.bool)

    p1 = controller.classify_and_kick_hit(EXPERT_LAYER_ID, dummy_mask)
    plan = controller.plan_misses(p1, EXPERT_LAYER_ID)

    global_unique_demand = sorted(_ek(k) for k in p1.demand)
    hit_ops = [
        {"expert": _ek(op.expert), "owner_rank": op.owner_rank, "slot": op.slot}
        for op in p1.hit_launch_info.hit_ops
    ]
    miss_per_rank = {
        str(r): [_ek(k) for k in p1.miss_per_rank.get(r, [])]
        for r in range(EP_SIZE)
    }
    misses = sorted(_ek(k) for v in p1.miss_per_rank.values() for k in v)
    # fetch_ops: PRESERVE controller order (do not sort away order bugs).
    fetch_ops = [
        {"expert": _ek(op.expert), "fetcher_rank": op.fetcher_rank,
         "dst_slot": op.dst_slot, "victim": _ek(op.victim_expert),
         "order": op.order}
        for op in plan.fetch_ops
    ]
    routing_map = {
        f"{k[0]}:{k[1]}": rank for k, rank in plan.expert_to_rank.items()
    }
    return {
        "global_unique_demand": global_unique_demand,
        "hit_ops": hit_ops,
        "misses": misses,
        "miss_per_rank": miss_per_rank,
        "fetch_ops": fetch_ops,
        "routing_map": routing_map,
        "shadow_after": _shadow(controller),
    }


def _seed_shadow(controller, shadow_before):
    """Isolated mode: seed each rank's slots in slot order so last_used /
    inserted_at reflect slot index (slot0 oldest)."""
    for r_str, slots in shadow_before.items():
        rc = controller.cache_view.per_rank[int(r_str)]
        for i, cell in enumerate(slots):
            if cell is not None:
                rc.apply(slot=i, evict=None,
                         insert=(int(cell[0]), int(cell[1])), demand_count=0)


# ============================================================
# field-by-field assertions + structural invariants
# ============================================================
def _assert_step(actual, expected, trace_step, mode):
    tag = f"[{mode} trace_step={trace_step}]"
    for field in ("global_unique_demand", "hit_ops", "misses",
                  "miss_per_rank", "fetch_ops", "routing_map", "shadow_after"):
        assert actual[field] == expected[field], (
            f"{tag} {field} mismatch:\n  actual  ={actual[field]}\n"
            f"  expected={expected[field]}")


def _assert_invariants(actual, summary, trace_step, mode):
    tag = f"[{mode} trace_step={trace_step}]"
    gud = len(actual["global_unique_demand"])
    hit = len(actual["hit_ops"])
    miss = len(actual["misses"])
    fetch = len(actual["fetch_ops"])
    evict = sum(1 for op in actual["fetch_ops"] if op["victim"] is not None)

    assert gud == hit + miss, f"{tag} gud {gud} != hit {hit} + miss {miss}"
    assert miss == fetch, f"{tag} miss {miss} != fetch {fetch}"
    assert evict == summary["evict_ops_count"], (
        f"{tag} evict {evict} != golden {summary['evict_ops_count']}")

    # duplicate_fetch == {}
    seen = {}
    for op in actual["fetch_ops"]:
        k = tuple(op["expert"])
        seen[k] = seen.get(k, 0) + 1
    dup_fetch = {f"{k[0]}:{k[1]}": c for k, c in seen.items() if c > 1}
    assert dup_fetch == {}, f"{tag} duplicate_fetch {dup_fetch}"

    # duplicate_resident_experts == {} (no expert resident in >1 rank/slot)
    resident = {}
    for r_str, slots in actual["shadow_after"].items():
        for cell in slots:
            if cell is not None:
                k = tuple(cell)
                resident[k] = resident.get(k, 0) + 1
    dup_res = {f"{k[0]}:{k[1]}": c for k, c in resident.items() if c > 1}
    assert dup_res == {}, f"{tag} duplicate_resident_experts {dup_res}"

    # match golden summary counts
    assert gud == summary["global_unique_demand_count"], f"{tag} gud count"
    assert hit == summary["hit_count"], f"{tag} hit count"
    assert miss == summary["miss_count"], f"{tag} miss count"
    assert fetch == summary["fetch_ops_count"], f"{tag} fetch count"


# ============================================================
# Sequential mode — one controller, cumulative steps 0..4
# ============================================================
def test_controller_golden_sequential():
    controller = _make_controller()
    prev_shadow_after = None
    for line in GOLD:
        ts = line["trace_step"]
        expected = line["expected"]

        # shadow_before chaining: current shadow must equal golden before.
        cur_shadow = _shadow(controller)
        assert cur_shadow == expected["shadow_before"], (
            f"[sequential trace_step={ts}] shadow_before mismatch:\n"
            f"  actual  ={cur_shadow}\n  expected={expected['shadow_before']}")
        if prev_shadow_after is not None:
            assert prev_shadow_after == expected["shadow_before"], (
                f"[sequential trace_step={ts}] prev.shadow_after != "
                f"this.shadow_before")

        actual = _run_step(controller, line)
        _assert_step(actual, expected, ts, "sequential")
        _assert_invariants(actual, line["expected"]["summary"], ts,
                           "sequential")

        prev_shadow_after = actual["shadow_after"]
        controller.verify_layer_end(EXPERT_LAYER_ID)  # bump layer boundary


# ============================================================
# Isolated mode — fresh controller per step, shadow seeded directly.
#
# IMPORTANT: under true-cumulative LRU the eviction victim is a function of
# cross-step recency (meta.last_used), which is NOT carried by shadow_before
# (a plain slot->expert map).  So a freshly-seeded cache cannot reproduce the
# victim/shadow_after of an EVICTION step — that information was lost.
# Isolated mode therefore verifies:
#   * always: the recency-INDEPENDENT fields (demand, hit_ops, misses,
#     miss_per_rank, routing_map) — these depend only on shadow_before contents
#     and the (deterministic) static owner policy.
#   * only on no-eviction steps (evict_ops_count == 0): fetch_ops + shadow_after
#     too, since an empty-slot fill picks first_empty_slot regardless of recency.
# ============================================================
_RECENCY_INDEPENDENT = (
    "global_unique_demand", "hit_ops", "misses", "miss_per_rank", "routing_map")


@pytest.mark.parametrize("idx", range(len(GOLD)))
def test_controller_golden_isolated(idx):
    line = GOLD[idx]
    ts = line["trace_step"]
    expected = line["expected"]

    controller = _make_controller()
    _seed_shadow(controller, expected["shadow_before"])
    assert _shadow(controller) == expected["shadow_before"], (
        f"[isolated trace_step={ts}] seeding failed")

    actual = _run_step(controller, line)

    for field in _RECENCY_INDEPENDENT:
        assert actual[field] == expected[field], (
            f"[isolated trace_step={ts}] {field} mismatch:\n"
            f"  actual  ={actual[field]}\n  expected={expected[field]}")

    if expected["summary"]["evict_ops_count"] == 0:
        for field in ("fetch_ops", "shadow_after"):
            assert actual[field] == expected[field], (
                f"[isolated trace_step={ts}] {field} mismatch:\n"
                f"  actual  ={actual[field]}\n  expected={expected[field]}")

    _assert_invariants(actual, line["expected"]["summary"], ts, "isolated")


if __name__ == "__main__":
    test_controller_golden_sequential()
    for i in range(len(GOLD)):
        test_controller_golden_isolated(i)
    print("controller golden trajectory: ALL PASS")
