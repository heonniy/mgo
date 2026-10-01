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
