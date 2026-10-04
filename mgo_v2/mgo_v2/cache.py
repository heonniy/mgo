from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .types import ExpertKey


@dataclass
class CacheEntry:
    key: ExpertKey
    slot: int
    last_used: int = 0
    admitted_at: int = 0


class RankCache:
    def __init__(self, rank: int, capacity: int):
        self.rank = rank
        self.capacity = capacity
        self.slots: list[Optional[ExpertKey]] = [None] * capacity
        self.entries: Dict[ExpertKey, CacheEntry] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def contains(self, key: ExpertKey) -> bool:
        return key in self.entries

    def slot_of(self, key: ExpertKey) -> int:
        return self.entries[key].slot

    def free_slot(self) -> Optional[int]:
        for i, key in enumerate(self.slots):
            if key is None:
                return i
        return None

    def keys(self) -> Iterable[ExpertKey]:
        return self.entries.keys()

    def touch(self, key: ExpertKey, tick: int) -> None:
        if key in self.entries:
            self.entries[key].last_used = tick

    def evict(self, key: ExpertKey) -> int:
        entry = self.entries.pop(key)
        self.slots[entry.slot] = None
        return entry.slot

    def admit(self, key: ExpertKey, slot: int, tick: int) -> None:
        if key in self.entries:
            self.touch(key, tick)
            return
        if self.slots[slot] is not None:
            raise RuntimeError(f"rank {self.rank} slot {slot} not empty")
        self.slots[slot] = key
        self.entries[key] = CacheEntry(key, slot, tick, tick)


class GlobalCacheState:
    """Single-copy expert residency shared by all controller policies."""

    def __init__(self, capacities: list[int]):
        self.ranks = [RankCache(r, cap) for r, cap in enumerate(capacities)]
        self.owner: Dict[ExpertKey, int] = {}
        self.tick = 0

    def next_tick(self) -> int:
        self.tick += 1
        return self.tick

    def owner_of(self, key: ExpertKey) -> Optional[int]:
        return self.owner.get(key)

    def resident(self, key: ExpertKey) -> bool:
        return key in self.owner

    def resident_layer(self, layer: int) -> set[int]:
        return {e for (l, e), _r in self.owner.items() if l == layer}

    def keys_on_rank(self, rank: int, layer: Optional[int] = None) -> list[ExpertKey]:
        keys = list(self.ranks[rank].keys())
        if layer is None:
            return keys
        return [k for k in keys if k[0] == layer]

    def touch(self, key: ExpertKey, tick: Optional[int] = None) -> None:
        rank = self.owner[key]
        self.ranks[rank].touch(key, self.tick if tick is None else tick)

    def place(self, rank: int, key: ExpertKey, slot: int, tick: int) -> None:
        if key in self.owner:
            raise RuntimeError(f"replication disabled: {key} already on rank {self.owner[key]}")
        self.ranks[rank].admit(key, slot, tick)
        self.owner[key] = rank

    def evict(self, key: ExpertKey) -> tuple[int, int]:
        rank = self.owner.pop(key)
        slot = self.ranks[rank].evict(key)
        return rank, slot

    def assert_consistent(self) -> None:
        seen: set[ExpertKey] = set()
        for rank_cache in self.ranks:
            for key, entry in rank_cache.entries.items():
                if key in seen:
                    raise AssertionError(f"replicated key {key}")
                seen.add(key)
                if rank_cache.slots[entry.slot] != key:
                    raise AssertionError("slot/key mismatch")
                if self.owner.get(key) != rank_cache.rank:
                    raise AssertionError("owner mismatch")
        if seen != set(self.owner):
            raise AssertionError("owner map drift")

@dataclass(frozen=True)
class SlotPromotion:
    rank: int
    key: int
    logical_slot: int
    prefetch_slot: int
    victim: int
    promoted_physical_slot: int
    recycled_physical_slot: int


class SlotArena:
    """One authority for MAIN logical state and C+P physical role mappings.

    `main` is the validated array controller; its arrays are retained by reference,
    never copied into a second residency dictionary. CUDA readiness lives elsewhere.
    """
    def __init__(self, main, prefetch_capacity):
        import numpy as np
        if prefetch_capacity < 0:
            raise ValueError('negative prefetch capacity')
        self.main = main
        self.prefetch_capacity = int(prefetch_capacity)
        self.main_physical = [np.arange(int(c), dtype=np.int32) for c in main.capacities]
        self.prefetch_physical = [np.arange(int(c), int(c)+prefetch_capacity, dtype=np.int32) for c in main.capacities]
        self.reservations = {}

    def reserve(self, rank, key, pfslot, target_layer):
        if self.main.owner[key] or key in self.reservations:
            raise ValueError('duplicate MAIN/PREFETCH key')
        if not 0 <= pfslot < self.prefetch_capacity:
            raise ValueError('prefetch capacity exceeded')
        if any(r == rank and p == pfslot for r,p,_ in self.reservations.values()):
            raise ValueError('prefetch slot already reserved')
        if key//128 != target_layer:
            raise ValueError('target layer mismatch')
        self.reservations[key] = (rank,pfslot,target_layer)
        return int(self.prefetch_physical[rank][pfslot])

    def promote(self, key, logical_slot, tick):
        import numpy as np
        from br_carep_cpu import place
        rank,pfslot,_ = self.reservations[key]
        m = self.main
        if m.owner[key]:
            raise ValueError('promotion key already MAIN')
        victim = int(m.slots[rank,logical_slot])
        physical = int(self.prefetch_physical[rank][pfslot])
        recycled = int(self.main_physical[rank][logical_slot])
        place(rank,key,logical_slot,False,tick,m.slots,m.owner,m.primary,m.last,m.seen,m.lost,m.birth,m.reuses,np.zeros(48,np.float64))
        self.main_physical[rank][logical_slot] = physical
        self.prefetch_physical[rank][pfslot] = recycled
        del self.reservations[key]
        return SlotPromotion(rank,key,logical_slot,pfslot,victim,physical,recycled)

    def discard(self, key):
        rank,pfslot,_ = self.reservations.pop(key)
        return rank,int(self.prefetch_physical[rank][pfslot]),key

    def assert_consistent(self):
        import numpy as np
        seen = set()
        for rank,cap in enumerate(self.main.capacities):
            physical = list(self.main_physical[rank])+list(self.prefetch_physical[rank])
            assert sorted(physical) == list(range(int(cap)+self.prefetch_capacity))
            assert len(self.main_physical[rank]) == cap
            assert len(self.prefetch_physical[rank]) == self.prefetch_capacity
            for key in self.main.slots[rank,:cap]:
                if key < 0:continue
                assert int(key) not in seen
                seen.add(int(key))
                assert self.main.owner[key] == 1 << rank and self.main.primary[key] == rank
        assert seen == set(np.flatnonzero(self.main.owner))
        assert not seen.intersection(self.reservations)
        used = set()
        for key,(rank,pfslot,target) in self.reservations.items():
            assert key//128 == target and 0 <= pfslot < self.prefetch_capacity
            assert (rank,pfslot) not in used
            used.add((rank,pfslot))
