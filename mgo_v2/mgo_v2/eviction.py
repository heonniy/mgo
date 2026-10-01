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
        if len(probs) >= self.window:
            # Earlier rows cannot survive this event's W-token history.
            latest = np.asarray(probs[-self.window:], dtype=np.float64)
            q.clear()
            q.extend(row.copy() for row in latest)
            self.sums[layer] = latest.sum(axis=0)
            return
        for row in probs:
            r = np.asarray(row, dtype=np.float64).copy()
            q.append(r)
            self.sums[layer] += r
            if len(q) > self.window:
                self.sums[layer] -= q.popleft()

    def score(self, layer: int, expert: int) -> float:
        n = max(1, len(self.rows[layer]))
        # The trace packer computes a float64 mean and stores float32 scores.
        # Match that rounding before ranking, so equal reference scores tie.
        return float(np.float32(self.sums[layer, expert] / n))


def _rank01(values: dict[ExpertKey, float]) -> dict[ExpertKey, float]:
    denominator = 2 * max(1, len(values) - 1)
    return {key: value / denominator for key, value in _midrank2(values).items()}


def _midrank2(values: dict[ExpertKey, float]) -> dict[ExpertKey, int]:
    if not values:
        return {}
    keys = sorted(values, key=lambda k: (values[k], k))
    if len(keys) == 1:
        return {keys[0]: 0}
    out: dict[ExpertKey, int] = {}
    i = 0
    while i < len(keys):
        j = i + 1
        while j < len(keys) and values[keys[j]] == values[keys[i]]:
            j += 1
        # Preserve ties using the validated simulator's percentile midrank.
        percentile = 2 * i + j - i - 1
        for key in keys[i:j]:
            out[key] = percentile
        i = j
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
        self.neighbors = similarity >= similarity_threshold
        for layer in range(similarity.shape[0]):
            np.fill_diagonal(self.neighbors[layer], True)

    def _represents(self, layer: int, resident: int, source: int) -> bool:
        return resident == source or (
            self.similarity[layer, source, resident] >= self.similarity_threshold
        )

    def coverage_damage(
        self, cache: GlobalCacheState, key: ExpertKey
    ) -> int:
        layer, victim = key
        residents = sorted(cache.resident_layer(layer))
        counts = self.neighbors[layer][:, residents].sum(axis=1)
        return int(np.count_nonzero(self.neighbors[layer, :, victim] & (counts <= self.k_min)))

    def choose(self, cache, rank, layer, pinned):
        cand = self.candidates(cache, rank, pinned)
        if not cand:
            raise RuntimeError(f"rank {rank}: no legal victim")

        gate = {k: self.history.score(k[0], k[1]) for k in cand}
        # Compute global source coverage once per candidate layer, rather
        # than rebuilding the same counts for every resident victim.
        residents_by_layer = {}
        for resident_layer, expert in cache.owner:
            residents_by_layer.setdefault(resident_layer, []).append(expert)
        damage = {}
        for candidate_layer in {k[0] for k in cand}:
            neighbors = self.neighbors[candidate_layer]
            counts = neighbors[:, residents_by_layer[candidate_layer]].sum(axis=1)
            losses = neighbors[counts <= self.k_min].sum(axis=0)
            for key in cand:
                if key[0] == candidate_layer:
                    damage[key] = float(losses[key[1]])
        # The common denominator cancels in argmin. Keep doubled integer
        # numerators: normalized floats can break exact score ties by 1 ULP
        # and choose a different victim before the LRU tie-break.
        gate_rank = _midrank2(gate)
        damage_rank = _midrank2(damage)

        return min(
            cand,
            key=lambda k: (
                gate_rank[k] + self.lam * damage_rank[k],
                cache.ranks[rank].entries[k].last_used,
                k,
            ),
        )
