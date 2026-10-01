"""Golden Test 2 — FakeArcher replay (moe_cache_golden README §2).

Replays the per-rank ``submit_plan`` tuples recorded in
``fake_archer_gold_replay.jsonl`` against a slot-level FakeArcherDispatcher and
verifies the *physical slot* bookkeeping exactly: commit order, the victim that
must occupy ``dst_slot`` right before each commit, old->new replacement, and
the final slot mapping.

This is independent of the controller's planning policy — it replays the tuples
the golden records, so it catches archer-side execution bugs:
  * hit pretending   — a hit tuple points at a slot that does not hold it
  * victim mismatch  — controller says victim X is in dst_slot but it is not
  * slot-level drift — expert *set* matches but slot *mapping* differs
"""
from __future__ import annotations

import copy
import json
import os

import pytest


GOLD_DIR = os.environ.get(
    "MOE_EP_GOLD_DIR", "/home/work/hyewon.lee/실험/moe_cache_golden")
GOLD_FILE = os.path.join(GOLD_DIR, "fake_archer_gold_replay.jsonl")

# Superseded 2026-06-03: golden moved to ``superseded/``; replaced by the
# import-pure oracle suite ``tests/ref/test_fake_archer_replay.py``.  Skip at
# module level if the old golden is gone (avoids a collection error).
if not os.path.exists(GOLD_FILE):
    pytest.skip("old golden moved to superseded/ — replaced by tests/ref/ "
                "fake-archer replay", allow_module_level=True)


def _load_gold():
    with open(GOLD_FILE) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


GOLD = _load_gold()


def _cell(x):
    """[l, e] -> (l, e); None -> None."""
    return None if x is None else (int(x[0]), int(x[1]))


def _norm_slots(slots):
    """internal [(l,e)|None,...] -> golden JSON [[l,e]|None,...]."""
    return [None if s is None else [s[0], s[1]] for s in slots]


# ============================================================
# FakeArcherDispatcher — slot-level commit replay (README §2)
# ============================================================
class FakeArcherDispatcher:
    def __init__(self, cap):
        self.cap = cap
        self.slot_to_key = [None] * cap
        self.events = []

    def seed_slots(self, slots):
        """slots: golden physical_slot_before list of [l,e]|None."""
        assert len(slots) == self.cap
        self.slot_to_key = [_cell(s) for s in slots]

    def submit_plan(self, gpu, hit_tuples, miss_tuples):
        # 1. hit validation — the slot really holds the claimed expert.
        for (l, e, slot) in hit_tuples:
            key = (l, e)
            assert self.slot_to_key[slot] == key, (
                f"hit pretending: slot {slot} holds "
                f"{self.slot_to_key[slot]}, hit tuple claims {key}")
            self.events.append(
                {"event": "hit_validate", "expert": [l, e], "slot": slot})

        # 2. replay misses by order, checking victim then committing old->new.
        for (l, e, dst, vl, ve, order) in sorted(miss_tuples, key=lambda t: t[5]):
            new = (l, e)
            if vl < 0:                       # empty-slot fill
                expected_victim = None
                assert self.slot_to_key[dst] is None, (
                    f"victim mismatch: dst_slot {dst} expected empty, "
                    f"holds {self.slot_to_key[dst]}")
            else:
                expected_victim = (vl, ve)
                assert self.slot_to_key[dst] == expected_victim, (
                    f"victim mismatch: dst_slot {dst} holds "
                    f"{self.slot_to_key[dst]}, plan says victim "
                    f"{expected_victim}")
            old = self.slot_to_key[dst]
            self.slot_to_key[dst] = new      # commit old -> new
            self.events.append({
                "event": "commit", "order": order, "slot": dst,
                "old": list(old) if old else None,
                "expected_victim": (
                    list(expected_victim) if expected_victim else None),
                "new": [new[0], new[1]],
            })

    def get_cached_slots(self, gpu):
        return [
            None if s is None else [i, s[0], s[1]]
            for i, s in enumerate(self.slot_to_key)
        ]

    def get_cached_experts(self, gpu):
        return [s for s in self.slot_to_key if s is not None]

    def commits(self):
        """Commit records in golden ``expected_commits`` shape (no event tag)."""
        return [
            {"order": e["order"], "slot": e["slot"], "old": e["old"],
             "expected_victim": e["expected_victim"], "new": e["new"]}
            for e in self.events if e["event"] == "commit"
        ]


# ============================================================
# per-trace_step replay test
# ============================================================
@pytest.mark.parametrize("idx", range(len(GOLD)))
def test_fake_archer_replay(idx):
    line = GOLD[idx]
    ts = line["trace_step"]
    exp = line["expected"]
    cap = len(line["expected"]["physical_slot_before"]["0"])

    for r_str, rank_exp in exp["per_rank"].items():
        tag = f"[trace_step={ts} rank={r_str}]"
        disp = FakeArcherDispatcher(cap)
        disp.seed_slots(rank_exp["slot_before"])

        spr = rank_exp["submit_plan_received"]
        disp.submit_plan(int(r_str), spr["hit_tuples"], spr["miss_tuples"])

        # exact commit comparison: order, slot, old, expected_victim, new
        assert disp.commits() == rank_exp["expected_commits"], (
            f"{tag} commits mismatch:\n  actual  ={disp.commits()}\n"
            f"  expected={rank_exp['expected_commits']}")

        # exact slot_after
        assert _norm_slots(disp.slot_to_key) == rank_exp["slot_after"], (
            f"{tag} slot_after mismatch:\n"
            f"  actual  ={_norm_slots(disp.slot_to_key)}\n"
            f"  expected={rank_exp['slot_after']}")

    # global physical_slot_after across ranks
    actual_phys = {}
    for r_str, rank_exp in exp["per_rank"].items():
        disp = FakeArcherDispatcher(cap)
        disp.seed_slots(rank_exp["slot_before"])
        spr = rank_exp["submit_plan_received"]
        disp.submit_plan(int(r_str), spr["hit_tuples"], spr["miss_tuples"])
        actual_phys[r_str] = _norm_slots(disp.slot_to_key)
    assert actual_phys == exp["physical_slot_after"], (
        f"[trace_step={ts}] physical_slot_after mismatch")


def test_fake_archer_slot_after_equals_physical():
    """README §7: FakeArcher slot_after == physical_slot_after (no drift)."""
    for line in GOLD:
        exp = line["expected"]
        for r_str, rank_exp in exp["per_rank"].items():
            assert rank_exp["slot_after"] == exp["physical_slot_after"][r_str]


# ============================================================
# negative tests — the replay must DETECT corruption (README §2 goals)
# ============================================================
def test_detects_hit_pretending():
    """A hit tuple pointing at a slot that does not hold it must raise."""
    disp = FakeArcherDispatcher(4)
    disp.seed_slots([[0, 0], [0, 4], None, None])
    with pytest.raises(AssertionError, match="hit pretending"):
        # claim E0 is at slot 1 (it is at slot 0)
        disp.submit_plan(0, hit_tuples=[[0, 0, 1]], miss_tuples=[])


def test_detects_victim_mismatch():
    """A miss whose dst_slot does not hold the claimed victim must raise."""
    disp = FakeArcherDispatcher(4)
    disp.seed_slots([[0, 0], [0, 4], None, None])
    with pytest.raises(AssertionError, match="victim mismatch"):
        # claim victim E9 in slot0, but slot0 holds E0
        disp.submit_plan(0, hit_tuples=[],
                         miss_tuples=[[0, 16, 0, 0, 9, 0]])


def test_detects_fill_into_occupied_slot():
    """An empty-fill (victim=-1) targeting an occupied slot must raise."""
    disp = FakeArcherDispatcher(4)
    disp.seed_slots([[0, 0], None, None, None])
    with pytest.raises(AssertionError, match="expected empty"):
        disp.submit_plan(0, hit_tuples=[],
                         miss_tuples=[[0, 16, 0, -1, -1, 0]])


if __name__ == "__main__":
    for i in range(len(GOLD)):
        test_fake_archer_replay(i)
    test_fake_archer_slot_after_equals_physical()
    print("fake archer replay: ALL PASS")
