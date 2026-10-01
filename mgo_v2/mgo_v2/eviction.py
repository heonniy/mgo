from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from .cache import GlobalCacheState
from .types import ExpertKey


class GateHistory:
    """Per-layer rolling window over full router probabilities."""

    def __init__(self, num_layers: int, num_experts: int, window: int = 128):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.window = window
        self.rows = [deque() for _ in range(num_layers)]
        self.sums = np.zeros((num_layers, num_experts), dtype=np.float64)

    def update(self, layer: int, probs: np.ndarray) -> None:
        if probs is None:
            return
        if probs.ndim != 2 or probs.shape[1] != self.num_experts:
            raise ValueError("probs must be [tokens, experts]")
        q = self.rows[layer]
        for row in probs:
            r = np.asarray(row, dtype=np.float64).copy()
            q.append(r)
            self.sums[layer] += r
            if len(q) > self.window:
                self.sums[layer] -= q.popleft()

    def score(self, layer: int, expert: int) -> float:
        n = max(1, len(self.rows[layer]))
        return float(self.sums[layer, expert] / n)


def _rank01(values: dict[ExpertKey, float]) -> dict[ExpertKey, float]:
    if not values:
        return {}
    keys = sorted(values, key=lambda k: (values[k], k))
    if len(keys) == 1:
        return {keys[0]: 0.0}
    out: dict[ExpertKey, float] = {}
    for i, key in enumerate(keys):
        out[key] = i / (len(keys) - 1)
    return out


class EvictionPolicy:
    name = "base"

    def choose(
        self,
        cache: GlobalCacheState,
        rank: int,
        layer: int,
        pinned: set[ExpertKey],
    ) -> ExpertKey:
        raise NotImplementedError

    @staticmethod
    def candidates(
        cache: GlobalCacheState, rank: int, pinned: set[ExpertKey]
    ) -> list[ExpertKey]:
        return [k for k in cache.keys_on_rank(rank) if k not in pinned]


class LRUEviction(EvictionPolicy):
    name = "lru"

    def choose(self, cache, rank, layer, pinned):
        cand = self.candidates(cache, rank, pinned)
        if not cand:
            raise RuntimeError(f"rank {rank}: no legal victim")
        return min(
            cand,
            key=lambda k: (
                cache.ranks[rank].entries[k].last_used,
                cache.ranks[rank].entries[k].admitted_at,
                k,
            ),
        )


class GateScoreEviction(EvictionPolicy):
    name = "gate"

    def __init__(self, history: GateHistory):
        self.history = history

    def choose(self, cache, rank, layer, pinned):
        cand = self.candidates(cache, rank, pinned)
        if not cand:
            raise RuntimeError(f"rank {rank}: no legal victim")
        return min(
            cand,
            key=lambda k: (
                self.history.score(k[0], k[1]),
                cache.ranks[rank].entries[k].last_used,
                k,
            ),
        )


class DiversityEviction(EvictionPolicy):
    name = "coverage"

    def __init__(
        self,
        history: GateHistory,
        similarity: np.ndarray,
        similarity_threshold: float = 0.65,
        lam: float = 2.0,
        k_min: int = 1,
    ):
        self.history = history
        self.similarity = similarity
        self.similarity_threshold = similarity_threshold
        self.lam = lam
        self.k_min = k_min

    def _represents(self, layer: int, resident: int, source: int) -> bool:
        return resident == source or (
            self.similarity[layer, source, resident] >= self.similarity_threshold
        )

    def coverage_damage(
        self, cache: GlobalCacheState, key: ExpertKey
    ) -> int:
        layer, victim = key
        residents = sorted(cache.resident_layer(layer))
        counts = np.zeros(self.similarity.shape[1], dtype=np.int32)
        for source in range(self.similarity.shape[1]):
            counts[source] = sum(
                1 for resident in residents if self._represents(layer, resident, source)
            )
        damage = 0
        for source in range(self.similarity.shape[1]):
            before = max(0, self.k_min - int(counts[source]))
            after_count = int(counts[source]) - int(
                self._represents(layer, victim, source)
            )
            after = max(0, self.k_min - after_count)
            damage += after - before
        return int(damage)

    def choose(self, cache, rank, layer, pinned):
        cand = self.candidates(cache, rank, pinned)
        if not cand:
            raise RuntimeError(f"rank {rank}: no legal victim")

        gate = {k: self.history.score(k[0], k[1]) for k in cand}
        damage = {k: float(self.coverage_damage(cache, k)) for k in cand}
        gate_rank = _rank01(gate)
        damage_rank = _rank01(damage)

        return min(
            cand,
            key=lambda k: (
                gate_rank[k] + self.lam * damage_rank[k],
                cache.ranks[rank].entries[k].last_used,
                k,
            ),
        )
