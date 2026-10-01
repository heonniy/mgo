from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ExpertKey = Tuple[int, int]  # (layer, expert)


@dataclass(frozen=True)
class LayerRoutes:
    layer: int
    origin_ranks: np.ndarray        # [T]
    selected_experts: np.ndarray    # [T, K]
    routing_weights: np.ndarray     # [T, K]
    full_router_probs: Optional[np.ndarray] = None  # [T, E]

    def __post_init__(self):
        if self.selected_experts.shape != self.routing_weights.shape:
            raise ValueError("selected_experts and routing_weights shape mismatch")
        if self.selected_experts.shape[0] != self.origin_ranks.shape[0]:
            raise ValueError("token count mismatch")


@dataclass
class SubstitutionResult:
    source_to_target: Dict[int, int]
    target_tier: Dict[int, str]
    exact_hits: set[int]
    protected_misses: set[int]
    low_misses: set[int]
    residual_exact_misses: set[int]

    def target_for(self, source: int) -> int:
        return self.source_to_target.get(source, source)


@dataclass
class AdmissionResult:
    expert_to_rank: Dict[int, int]
    quotas: List[int]
    policy_name: str


@dataclass
class LocalExecPlan:
    hit_ops: List[Tuple[int, int, int]] = field(default_factory=list)
    miss_ops: List[Tuple[int, int, int, int, int, int]] = field(default_factory=list)


@dataclass
class LayerPlan:
    layer: int
    substitution: SubstitutionResult
    admission: AdmissionResult
    owner_by_expert: Dict[int, int]
    local_exec: Dict[int, LocalExecPlan]
    effective_token_routes: List[Dict[int, float]]
    pinned_keys: set[ExpertKey]
