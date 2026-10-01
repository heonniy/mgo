"""Slot-based cache view — new architecture (2026-05-27).

Replaces the OrderedDict-based GlobalCacheView with a fixed-size slot array
per rank.  Key changes vs the legacy view:

  * **Slot-as-physical-resource**: cap_per_rank slots indexed [0..cap-1],
    each holding zero or one ExpertKey.  Evict/install is a single
    ``apply(slot, evict, insert)`` atomic op — never out of sync with the
    eviction policy's metadata.

  * **No LRU OrderedDict reordering**: eviction policy reads ``meta[slot]``
    (last_used / freq / inserted_at).  Sequential planning + immediate
    ``apply`` after each victim selection means metadata is always fresh
    for the next pick — no "tail move" cleanup needed.

  * **No protected_keys set in the planner API**: the I5 invariant (a
    slot's hit-compute read finishes before its fetch-replace write) makes
    pinning the current layer's hits unnecessary.  Selecting a hit as a
    victim is safe: archer's compute stream consumes the slot's data
    before the copy stream overwrites it.

  * **Verify, not reconcile, at layer end**: shadow ↔ archer_physical is
    expected byte-identical under I1-I3.  ``verify_against_archer`` asserts;
    no silent re-construction.

NB: ``cap_per_rank`` here is the slot count, NOT a byte budget.  Byte
limits are still enforced by archer's ``cache_sizes_``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


ExpertKey = Tuple[int, int]  # (layer_id, expert_id)


@dataclass
class SlotMeta:
    """Per-slot metadata used by eviction policies (LRU / LFU / Priority).

    Attributes:
        expert        : (layer_id, expert_id) currently held by this slot
        inserted_at   : layer_seq when this expert was placed in this slot
        last_used     : layer_seq when this expert was last consumed
        freq          : ACCESS COUNT — # times this expert was demanded while
                        resident (classic LFU, +1 per layer-use, token-agnostic;
                        matches MoE-Infinity entry.visit).  Reset on re-insert.
    """
    expert: ExpertKey
    inserted_at: int
    last_used: int
    freq: int


class RankSlotCache:
    """One rank's slot-array cache.

    Invariants (asserted at every ``apply``):
      I1. ``slots[i] is None`` iff slot i is empty
      I2. ``slots[i] is meta[i].expert`` when not None
      I3. ``expert_to_slot[e] == i`` iff ``slots[i] == e``
      I4. ``len(slots) == len(meta) == cap``
    """

    def __init__(self, rank: int, cap: int):
        self.rank = rank
        self.cap = int(cap)
        self.slots: List[Optional[ExpertKey]] = [None] * self.cap
        self.expert_to_slot: Dict[ExpertKey, int] = {}
        self.meta: List[Optional[SlotMeta]] = [None] * self.cap
        # ``layer_seq`` — layer 단위 카운터, ``bump_layer`` 가 ``inserted_at``
        # 의 unit 으로 사용.
        self.layer_seq: int = 0
        # ``touch_counter`` — apply / touch_use 마다 monotonically 증가하는
        # **sub-layer** 카운터.  ``last_used`` 의 unit.  같은 layer 안에서도
        # 순서가 의미 있어 sequential plan 의 "방금 install 한 expert 가
        # 다음 iter 의 victim 안 됨" invariant 자연 보장 (디자인 §2 Phase 2
        # apply 의 "막 들어왔으니 — 다음 iter에서 또 victim 안 되게" 코멘트).
        self.touch_counter: int = 0

    # ---------- queries (read-only) ----------

    def is_resident(self, key: ExpertKey) -> bool:
        return key in self.expert_to_slot

    def find_slot(self, key: ExpertKey) -> Optional[int]:
        return self.expert_to_slot.get(key)

    def free_slots(self) -> int:
        return sum(1 for s in self.slots if s is None)

    def first_empty_slot(self) -> Optional[int]:
        for i, s in enumerate(self.slots):
            if s is None:
                return i
        return None

    def occupied_keys(self) -> Set[ExpertKey]:
        return set(self.expert_to_slot.keys())

    def occupied_slots(self) -> List[Tuple[int, SlotMeta]]:
        """Iterate over (slot_idx, meta) for occupied slots, in slot order."""
        return [(i, m) for i, m in enumerate(self.meta) if m is not None]

    # ---------- mutations ----------

    def apply(
        self,
        slot: int,
        evict: Optional[ExpertKey],
        insert: Optional[ExpertKey],
        demand_count: int = 0,
    ) -> None:
        """Atomic slot replace.

        Three legal cases:
          * evict=A, insert=B : slot[i]==A → slot[i]:=B (full replace)
          * evict=None, insert=B : slot[i] was empty → slot[i]:=B (fill)
          * evict=A, insert=None : slot[i]==A → slot[i]:=None (pure evict)

        ``demand_count`` is accepted for API stability but NO LONGER weights
        ``freq`` — classic access-count LFU sets ``freq=1`` at insert (one use).
        """
        if not (0 <= slot < self.cap):
            raise ValueError(
                f"apply: slot={slot} out of range [0, {self.cap})")

        # I1/I3 pre-check.
        cur = self.slots[slot]
        if evict is None:
            if cur is not None:
                raise AssertionError(
                    f"apply: slot {slot} is occupied by {cur}, "
                    f"expected empty (evict=None)")
        else:
            if cur != evict:
                raise AssertionError(
                    f"apply: slot {slot} holds {cur}, expected {evict}")
            if self.expert_to_slot.get(evict) != slot:
                raise AssertionError(
                    f"apply: expert_to_slot[{evict}]="
                    f"{self.expert_to_slot.get(evict)} != slot {slot}")
            del self.expert_to_slot[evict]

        # Insert phase.
        if insert is None:
            self.slots[slot] = None
            self.meta[slot] = None
            return

        if insert in self.expert_to_slot:
            raise AssertionError(
                f"apply: insert={insert} already at slot "
                f"{self.expert_to_slot[insert]}, cannot place at {slot}")

        self.slots[slot] = insert
        self.expert_to_slot[insert] = slot
        self.touch_counter += 1
        self.meta[slot] = SlotMeta(
            expert=insert,
            inserted_at=self.layer_seq,
            last_used=self.touch_counter,
            freq=1,                      # access-count LFU: first use
        )

    def touch_use(self, key: ExpertKey, demand_count: int = 0) -> None:
        """Mark a resident expert as used in the current layer.

        Updates ``last_used`` (for LRU) and increments ``freq`` by 1 (access-count
        LFU — one demand event, token-agnostic).  ``demand_count`` accepted for
        API stability but unused.  Does NOT change the slot's physical position.
        """
        slot = self.expert_to_slot.get(key)
        if slot is None:
            return
        m = self.meta[slot]
        if m is None:
            return
        self.touch_counter += 1
        m.last_used = self.touch_counter
        m.freq += 1                      # access-count LFU: +1 per use

    def bump_layer(self) -> None:
        """Called by controller at the end of each layer; advances layer_seq."""
        self.layer_seq += 1

    # ---------- archer sync ----------

    def verify_against_archer(
        self,
        archer_set: Iterable[ExpertKey],
    ) -> Tuple[int, Set[ExpertKey], Set[ExpertKey]]:
        """Compare shadow vs archer's actual cached set.

        Returns ``(drift_count, missing_in_archer, extra_in_archer)``.
        Under I1-I3 this should always be ``(0, set(), set())``.
        Non-zero is a correctness bug — caller should raise.
        """
        archer = set(archer_set)
        shadow = self.occupied_keys()
        missing = shadow - archer
        extra = archer - shadow
        return len(missing) + len(extra), missing, extra

    # ---------- debugging ----------

    def snapshot(self) -> Dict:
        """Dictionary snapshot for tests / debug.  Avoid in hot path."""
        return {
            "rank": self.rank,
            "cap": self.cap,
            "free": self.free_slots(),
            "occupied": [
                {"slot": i, "expert": m.expert,
                 "inserted_at": m.inserted_at, "last_used": m.last_used,
                 "freq": m.freq}
                for i, m in enumerate(self.meta) if m is not None
            ],
        }


class GlobalSlotCacheView:
    """Per-EP-group snapshot of slot caches.  One ``RankSlotCache`` per rank.

    Replicated byte-identical across all controllers under I1-I3.  All
    decisions read this view; all mutations route through
    ``per_rank[r].apply(...)``.
    """

    def __init__(self, ep_size: int, cap_per_rank: int,
                 num_layers: int, num_experts: int):
        self.ep_size = int(ep_size)
        self.cap_per_rank = int(cap_per_rank)
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.per_rank: List[RankSlotCache] = [
            RankSlotCache(rank=r, cap=cap_per_rank)
            for r in range(self.ep_size)
        ]

    # ---------- global queries ----------

    def locate(self, key: ExpertKey) -> Optional[int]:
        """Lowest rank that currently has this expert resident, or None."""
        for r, c in enumerate(self.per_rank):
            if c.is_resident(key):
                return r
        return None

    def locate_all(self, key: ExpertKey) -> List[int]:
        return [r for r, c in enumerate(self.per_rank) if c.is_resident(key)]

    def total_slots_used(self) -> int:
        return sum(c.cap - c.free_slots() for c in self.per_rank)

    def global_unique_experts(self) -> int:
        seen: Set[ExpertKey] = set()
        for c in self.per_rank:
            seen.update(c.occupied_keys())
        return len(seen)

    # ---------- layer boundary ----------

    def bump_all_layers(self) -> None:
        for c in self.per_rank:
            c.bump_layer()

    # ---------- archer verify ----------

    def verify_against_archer(
        self,
        per_rank_archer: List[Iterable[ExpertKey]],
    ) -> Dict[int, Tuple[int, Set[ExpertKey], Set[ExpertKey]]]:
        """For each rank, verify shadow vs archer. Returns drift report.

        Caller decides whether to raise (under I1-I3 all drifts are bugs).
        """
        out = {}
        for r, archer in enumerate(per_rank_archer):
            out[r] = self.per_rank[r].verify_against_archer(archer)
        return out

    # ---------- debug snapshot ----------

    def snapshot(self) -> Dict:
        return {
            "ep_size": self.ep_size,
            "cap_per_rank": self.cap_per_rank,
            "per_rank": [c.snapshot() for c in self.per_rank],
        }
