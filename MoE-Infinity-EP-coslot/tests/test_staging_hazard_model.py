"""Golden Test 4 (model) — staging-vs-direct decision + event ordering.

Section 4 of the moe_cache_golden README targets the REAL Archer dispatcher,
which needs two C++ debug APIs that do not yet exist (``get_debug_events`` and a
slot-level ``get_cached_slots``).  Rather than rebuild the shared CUDA
extension (other live experiments depend on it), this test models the
staging/direct *decision rule* and event sequence in pure Python and validates
it against ``real_archer_gold_execution.jsonl``.

It is both a correctness check of the golden's section-4 logic and an
executable spec for the C++ debug events to be added later.  The decision rule
(matching expert_dispatcher.cpp StartNextFetch):

    miss with empty dst_slot or non-computing victim  -> DIRECT
    miss whose dst_slot holds a victim being computed  -> STAGING
        (victim slot is a current-layer hit; overwrite must wait for the GEMM)

Key hazard invariant: a staging commit must NOT appear before
``wait_victim_compute_done``.
"""
from __future__ import annotations

import json
import os

import pytest


GOLD_DIR = os.environ.get(
    "MOE_EP_GOLD_DIR", "/home/work/hyewon.lee/실험/moe_cache_golden")
GOLD_FILE = os.path.join(GOLD_DIR, "real_archer_gold_execution.jsonl")

# Superseded 2026-06-03: golden moved to ``superseded/``.  The staging/direct
# decision logic is now covered by the import-pure oracle suite
# ``tests/ref/test_fake_archer_replay.py`` + ``test_stream_overlap.py``.
if not os.path.exists(GOLD_FILE):
    pytest.skip("old golden moved to superseded/ — replaced by tests/ref/ "
                "fake-archer + stream-overlap suite", allow_module_level=True)


def _load_gold():
    with open(GOLD_FILE) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


GOLD = _load_gold()


def _ek(x):
    return None if x is None else [int(x[0]), int(x[1])]


# ============================================================
# StagingModelArcher — pure-Python model of the dispatcher fetch path
# ============================================================
class StagingModelArcher:
    def __init__(self, cap):
        self.cap = cap
        self.slots = [None] * cap
        self.events = []
        self.direct = 0
        self.staging = 0

    def seed(self, slots):
        self.slots = [None if s is None else (int(s[0]), int(s[1]))
                      for s in slots]

    def submit_plan(self, gpu, hit_tuples, miss_tuples):
        # slots whose resident expert is being computed this layer (hits)
        computing_slots = {slot for (_l, _e, slot) in hit_tuples}
        for (l, e, slot) in hit_tuples:
            self.events.append(
                {"event": "hit_task_created", "expert": [l, e], "slot": slot})

        for (l, e, dst, vl, ve, order) in sorted(miss_tuples, key=lambda t: t[5]):
            victim = None if vl < 0 else [vl, ve]
            # STAGING iff overwriting a slot whose victim is mid-GEMM.
            staging = (vl >= 0) and (dst in computing_slots)
            mode = "staging" if staging else "direct"
            self.events.append({
                "event": "fetch_begin", "order": order, "expert": [l, e],
                "dst_slot": dst, "victim": victim, "mode": mode})
            if staging:
                self.staging += 1
                self.events += [
                    {"event": "staging_h2d_begin", "order": order,
                     "expert": [l, e], "staging_slot": "reserved_cap_slot"},
                    {"event": "staging_h2d_done", "order": order,
                     "expert": [l, e], "staging_slot": "reserved_cap_slot"},
                    {"event": "wait_victim_compute_done", "victim": victim,
                     "slot": dst},
                    {"event": "staging_d2d_begin", "order": order,
                     "expert": [l, e], "from_slot": "reserved_cap_slot",
                     "dst_slot": dst},
                    {"event": "staging_d2d_done", "order": order,
                     "expert": [l, e], "dst_slot": dst},
                    {"event": "commit", "order": order, "slot": dst,
                     "old": victim, "new": [l, e], "mode": "staging"},
                ]
            else:
                self.direct += 1
                self.events += [
                    {"event": "direct_h2d_begin", "order": order,
                     "expert": [l, e], "dst_slot": dst},
                    {"event": "direct_h2d_done", "order": order,
                     "expert": [l, e], "dst_slot": dst},
                    {"event": "commit", "order": order, "slot": dst,
                     "old": victim, "new": [l, e], "mode": "direct"},
                ]
            self.slots[dst] = (l, e)

    def slot_after(self):
        return [None if s is None else [s[0], s[1]] for s in self.slots]


def _is_subsequence(sub, full):
    """Every dict in `sub` appears in `full` in the same relative order."""
    it = iter(full)
    return all(any(ev == s for ev in it) for s in sub)


def _run_rank(rank_exp, cap):
    a = StagingModelArcher(cap)
    a.seed(rank_exp["slot_before"]) if "slot_before" in rank_exp else None
    spr = rank_exp["submit_plan_received"]
    a.submit_plan(0, spr["hit_tuples"], spr["miss_tuples"])
    return a


# ============================================================
# per-case validation
# ============================================================
@pytest.mark.parametrize("idx", range(len(GOLD)))
def test_staging_model_matches_golden(idx):
    case = GOLD[idx]
    cid = case["case_id"]
    exp = case["expected"]
    phys_before = exp.get("physical_slot_before")

    tot_direct = 0
    tot_staging = 0
    for r_str, rank_exp in exp["per_rank"].items():
        tag = f"[{cid} rank={r_str}]"
        cap = (len(phys_before[r_str]) if phys_before
               else len(rank_exp["slot_before"]))
        a = StagingModelArcher(cap)
        if phys_before:
            a.seed(phys_before[r_str])
        else:
            a.seed(rank_exp["slot_before"])
        spr = rank_exp["submit_plan_received"]
        a.submit_plan(int(r_str), spr["hit_tuples"], spr["miss_tuples"])

        # 1. golden events are an ordered subsequence of the model events.
        assert _is_subsequence(rank_exp["events"], a.events), (
            f"{tag} golden events not a subsequence of model events:\n"
            f"  golden={rank_exp['events']}\n  model ={a.events}")

        # 2. counters match per-rank deltas.
        c = rank_exp["counters"]
        assert a.direct == c["direct_fetch_count_delta"], (
            f"{tag} direct {a.direct} != {c['direct_fetch_count_delta']}")
        assert a.staging == c["staging_fetch_count_delta"], (
            f"{tag} staging {a.staging} != {c['staging_fetch_count_delta']}")

        # 3. final slot mapping matches.
        assert a.slot_after() == rank_exp["slot_after"], (
            f"{tag} slot_after {a.slot_after()} != {rank_exp['slot_after']}")

        # 4. hazard invariant: every staging commit comes AFTER its
        #    wait_victim_compute_done.
        ev = a.events
        for i, e in enumerate(ev):
            if e["event"] == "commit" and e.get("mode") == "staging":
                waits = [j for j, w in enumerate(ev)
                         if w["event"] == "wait_victim_compute_done"
                         and j < i]
                assert waits, (
                    f"{tag} staging commit at {i} has no preceding "
                    f"wait_victim_compute_done")

        tot_direct += a.direct
        tot_staging += a.staging

    # 5. global counter totals.
    gs = exp["global_summary"]
    assert tot_direct == gs.get("total_direct_fetch_delta"), (
        f"[{cid}] total direct {tot_direct} != {gs.get('total_direct_fetch_delta')}")
    assert tot_staging == gs.get("total_staging_fetch_delta"), (
        f"[{cid}] total staging {tot_staging} != {gs.get('total_staging_fetch_delta')}")

    # 6. physical_slot_after (archer_only cases carry it).
    if "physical_slot_after" in exp:
        actual = {}
        for r_str, rank_exp in exp["per_rank"].items():
            cap = len(phys_before[r_str])
            a = StagingModelArcher(cap)
            a.seed(phys_before[r_str])
            spr = rank_exp["submit_plan_received"]
            a.submit_plan(int(r_str), spr["hit_tuples"], spr["miss_tuples"])
            actual[r_str] = a.slot_after()
        assert actual == exp["physical_slot_after"], (
            f"[{cid}] physical_slot_after mismatch:\n  actual={actual}\n"
            f"  expected={exp['physical_slot_after']}")


def test_staging_hazard_rank_modes():
    """README §7 acceptance: staging case uses staging on rank0 and direct on
    rank1/rank2; rank3 has no fetch."""
    case = next(c for c in GOLD if c["case_id"] == "archer_only_mixed_staging_hazard")
    pr = case["expected"]["per_rank"]

    def mode_of(r):
        a = _run_rank({**pr[r], "slot_before": case["expected"]
                       ["physical_slot_before"][r]},
                      len(case["expected"]["physical_slot_before"][r]))
        fb = [e for e in a.events if e["event"] == "fetch_begin"]
        return [e["mode"] for e in fb]

    assert mode_of("0") == ["staging"], "rank0 must use staging"
    assert mode_of("1") == ["direct"], "rank1 must use direct"
    assert mode_of("2") == ["direct"], "rank2 must use direct"
    assert mode_of("3") == [], "rank3 must have no fetch"


if __name__ == "__main__":
    for i in range(len(GOLD)):
        test_staging_model_matches_golden(i)
    test_staging_hazard_rank_modes()
    print("staging hazard model: ALL PASS")
