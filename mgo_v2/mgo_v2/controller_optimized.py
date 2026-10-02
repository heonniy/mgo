"""Opt-in decision-equivalent Coverage implementation; baseline classes stay intact."""
from __future__ import annotations

import numpy as np

from .cache import GlobalCacheState
from .eviction import DiversityEviction, GateHistory


class IndexedCache(GlobalCacheState):
    """The original authoritative cache plus incrementally maintained read views."""
    def __init__(self, capacities, layers, experts, debug=False):
        super().__init__(capacities)
        self.num_layers, self.num_experts = layers, experts
        self.debug = debug
        self.slot_keys = [np.full(n, -1, dtype=np.int64) for n in capacities]
        self.slot_used = [np.zeros(n, dtype=np.int64) for n in capacities]
        self.layer_residents = [set() for _ in range(layers)]
        self.layer_views = [frozenset() for _ in range(layers)]
        self.on_change = None

    def place(self, rank, key, slot, tick):
        super().place(rank, key, slot, tick)
        layer, expert = key
        self.slot_keys[rank][slot] = layer * self.num_experts + expert
        self.slot_used[rank][slot] = tick
        self.layer_residents[layer].add(expert)
        self.layer_views[layer] = None
        if self.on_change is not None:
            self.on_change(key, 1)

    def evict(self, key):
        rank, slot = super().evict(key)
        self.slot_keys[rank][slot] = -1
        self.slot_used[rank][slot] = 0
        self.layer_residents[key[0]].remove(key[1])
        self.layer_views[key[0]] = None
        if self.on_change is not None:
            self.on_change(key, -1)
        return rank, slot

    def touch(self, key, tick=None):
        super().touch(key, tick)
        rank = self.owner[key]
        self.slot_used[rank][self.ranks[rank].slot_of(key)] = self.tick if tick is None else tick

    def resident_layer(self, layer):
        if self.layer_views[layer] is None:
            self.layer_views[layer] = frozenset(self.layer_residents[layer])
        return self.layer_views[layer]

    def assert_consistent(self):
        if self.debug:
            self.validate_views()

    def validate_views(self):
        GlobalCacheState.assert_consistent(self)
        for rank, state in enumerate(self.ranks):
            expected = [key[0] * self.num_experts + key[1] if key is not None else -1
                        for key in state.slots]
            np.testing.assert_array_equal(self.slot_keys[rank], expected)
            for key, entry in state.entries.items():
                assert self.slot_used[rank][entry.slot] == entry.last_used
        for layer in range(self.num_layers):
            expected = {e for l, e in self.owner if l == layer}
            assert self.layer_residents[layer] == expected
            if self.layer_views[layer] is not None:
                assert self.layer_views[layer] == expected


class IndexedHistory(GateHistory):
    def __init__(self, layers, experts, window):
        super().__init__(layers, experts, window)
        self.scores = np.zeros((layers, experts), dtype=np.float32)

    def update(self, layer, probs):
        super().update(layer, probs)
        if probs is not None:
            self.scores[layer] = (self.sums[layer] / max(1, len(self.rows[layer]))).astype(np.float32)


def midrank2_array(values):
    """Exact doubled midranks, including singleton and tied groups."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (2 * starts + counts - 1)[inverse]


class IndexedCoverage(DiversityEviction):
    def __init__(self, history, similarity, cache, **kwargs):
        super().__init__(history, similarity, **kwargs)
        if cache.owner:
            raise ValueError('install indexed Coverage before the first admission')
        self.cache = cache
        self.counts = np.zeros(self.neighbors.shape[:2], dtype=np.int32)
        self.losses = self.neighbors.sum(axis=1)
        self.dirty = set()
        cache.on_change = self.residency_changed

    def residency_changed(self, key, direction):
        layer, expert = key
        self.counts[layer] += direction * self.neighbors[layer, :, expert]
        self.dirty.add(layer)

    def _sync_coverage(self, cache):
        if cache is not self.cache:
            raise ValueError('indexed Coverage is bound to its authoritative cache')
        for layer in self.dirty:
            self.losses[layer] = self.neighbors[layer][self.counts[layer] <= self.k_min].sum(axis=0)
        self.dirty.clear()

    def coverage_damage(self, cache, key):
        self._sync_coverage(cache)
        return int(self.losses[key])

    def candidate_slots(self, cache, rank, pinned):
        mask = cache.slot_keys[rank] >= 0
        # Pinning changes after every admission. Mark its few keys directly;
        # no full Python cache-key list or candidate tuple list is rebuilt.
        for key in pinned:
            if cache.owner_of(key) == rank:
                mask[cache.ranks[rank].slot_of(key)] = False
        return np.flatnonzero(mask)

    def choose(self, cache, rank, layer, pinned):
        slots = self.candidate_slots(cache, rank, pinned)
        if not len(slots):
            raise RuntimeError(f'rank {rank}: no legal victim')
        flat = cache.slot_keys[rank][slots]
        gate = self.history.scores.ravel()[flat]
        self._sync_coverage(cache)
        damage = self.losses.ravel()[flat]
        score = midrank2_array(gate) + self.lam * midrank2_array(damage)
        used = cache.slot_used[rank][slots]
        index = np.lexsort((flat, used, score))[0]
        return divmod(int(flat[index]), cache.num_experts)

    def validate_coverage(self):
        counts = np.zeros_like(self.counts)
        for layer, expert in self.cache.owner:
            counts[layer] += self.neighbors[layer, :, expert]
        np.testing.assert_array_equal(self.counts, counts)
        for layer in range(len(counts)):
            if layer in self.dirty:
                continue  # Pending recomputation is part of the next timed plan.
            expected = self.neighbors[layer][counts[layer] <= self.k_min].sum(axis=0)
            np.testing.assert_array_equal(self.losses[layer], expected)


def enable_local_optimization(controller, debug=False):
    """Opt in on a fresh empty controller; never changes an admission policy."""
    if controller.cache.owner or controller.cache.tick or any(controller.history.rows):
        raise ValueError('local optimization must be installed on a fresh controller')
    if not isinstance(controller.eviction, DiversityEviction):
        raise ValueError('this measured optimization is limited to Coverage')
    c = controller.config
    controller.cache = IndexedCache(c.per_rank_slots(), c.num_layers, c.num_experts, debug)
    controller.history = IndexedHistory(c.num_layers, c.num_experts, c.gate_window)
    controller.eviction = IndexedCoverage(controller.history, controller.similarity, controller.cache,
        similarity_threshold=c.similarity_threshold, lam=c.coverage_lambda, k_min=c.coverage_k)
    return controller
