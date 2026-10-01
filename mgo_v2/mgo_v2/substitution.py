from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

from .cache import GlobalCacheState
from .types import LayerRoutes, SubstitutionResult


class SubstitutionPolicy:
    """Expert-level cache-aware substitution.

    A missed expert is protected as a whole if any route to it has gate weight
    >= gate_threshold. Low-importance missed experts first reuse active exact
    hits / protected exact misses, then any other safe resident expert.
    """

    def __init__(
        self,
        similarity: np.ndarray,
        gate_threshold: float = 0.20,
        similarity_threshold: float = 0.65,
    ):
        if similarity.ndim != 3:
            raise ValueError("similarity must be [layers, experts, experts]")
        self.similarity = similarity
        self.gate_threshold = gate_threshold
        # The validated simulator stores both scores and its threshold as
        # float32. Preserve an exact boundary value such as float32(.65).
        self.similarity_threshold = float(np.asarray(similarity_threshold, dtype=similarity.dtype))

    def _best(self, layer: int, source: int, candidates: Iterable[int]) -> int | None:
        best = None
        best_score = -np.inf
        for cand in candidates:
            if cand == source:
                continue
            score = float(self.similarity[layer, source, cand])
            if score < self.similarity_threshold:
                continue
            if score > best_score or (score == best_score and (best is None or cand < best)):
                best = cand
                best_score = score
        return best

    def decide(self, routes: LayerRoutes, cache: GlobalCacheState) -> SubstitutionResult:
        layer = routes.layer
        weights_by_source: Dict[int, List[float]] = defaultdict(list)
        for experts, weights in zip(routes.selected_experts, routes.routing_weights):
            for expert, weight in zip(experts.tolist(), weights.tolist()):
                weights_by_source[int(expert)].append(float(weight))

        active_sources = set(weights_by_source)
        exact_hits = {
            e for e in active_sources if cache.resident((layer, e))
        }
        missed = active_sources - exact_hits
        protected_misses = {
            e
            for e in missed
            if max(weights_by_source[e], default=0.0) >= self.gate_threshold
        }
        low_misses = missed - protected_misses

        # Tier-1 anchors are already-paid execution: exact hits and protected
        # misses that must be fetched/executed exactly.
        tier1 = exact_hits | protected_misses
        resident = cache.resident_layer(layer)

        source_to_target: Dict[int, int] = {}
        target_tier: Dict[int, str] = {}

        for source in sorted(low_misses):
            target = self._best(layer, source, tier1)
            if target is not None:
                source_to_target[source] = target
                target_tier[source] = (
                    "active_exact_hit" if target in exact_hits else "protected_exact_miss"
                )
                continue

            target = self._best(layer, source, resident)
            if target is not None:
                source_to_target[source] = target
                target_tier[source] = "inactive_resident"

        residual_exact = missed - set(source_to_target)
        return SubstitutionResult(
            source_to_target=source_to_target,
            target_tier=target_tier,
            exact_hits=exact_hits,
            protected_misses=protected_misses,
            low_misses=low_misses,
            residual_exact_misses=residual_exact,
        )


def merge_effective_routes(
    routes: LayerRoutes, decision: SubstitutionResult
) -> list[dict[int, float]]:
    """Merge duplicate targets after source->target remapping per token."""
    out: list[dict[int, float]] = []
    for experts, weights in zip(routes.selected_experts, routes.routing_weights):
        merged: dict[int, float] = {}
        for source, weight in zip(experts.tolist(), weights.tolist()):
            target = decision.target_for(int(source))
            merged[target] = merged.get(target, 0.0) + float(weight)
        out.append(merged)
    return out
