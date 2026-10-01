from __future__ import annotations

from typing import Dict

import numpy as np

from .admission import (
    AdmissionContext,
    BalancedRandomAdmission,
    GreedyCurrentAdmission,
    GreedyPathAdmission,
    HungarianAdmission,
    SwapRefinedAdmission,
)
from .affinity import AffinityTables
from .cache import GlobalCacheState
from .config import RuntimeConfig
from .eviction import DiversityEviction, GateHistory, GateScoreEviction, LRUEviction
from .substitution import SubstitutionPolicy, merge_effective_routes
from .types import LayerPlan, LayerRoutes, LocalExecPlan


class GlobalExpertController:
    """The only authority for logical expert residency.

    The controller is deterministic and is intended to be replicated on every
    rank after all ranks have the same global routing metadata.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        similarity: np.ndarray,
        affinity: AffinityTables | None = None,
    ):
        self.config = config
        if similarity.shape != (config.num_layers, config.num_experts, config.num_experts):
            raise ValueError("similarity dimensions do not match the model")
        if not np.isfinite(similarity).all():
            raise ValueError("similarity contains nonfinite values")
        needs_same = config.admission in {"hungarian_same", "hungarian_same_path", "hungarian_swap"}
        needs_path = config.admission in {"greedy_path", "hungarian_same_path", "hungarian_swap"}
        if needs_same and (affinity is None or affinity.same_layer is None):
            raise ValueError(f"{config.admission} requires same_layer affinity")
        if needs_path and (affinity is None or affinity.path is None):
            raise ValueError(f"{config.admission} requires path affinity")
        if affinity is not None:
            for name, table, layers in (("same_layer", affinity.same_layer, config.num_layers),
                                        ("path", affinity.path, config.num_layers - 1)):
                if table is not None and (table.shape != (layers, config.num_experts, config.num_experts)
                                          or not np.isfinite(table).all()):
                    raise ValueError(f"invalid {name} affinity dimensions or nonfinite values")
        self.similarity = similarity
        self.affinity = affinity
        self.cache = GlobalCacheState(config.per_rank_slots())
        self.history = GateHistory(
            config.num_layers, config.num_experts, config.gate_window
        )
        self.substitution = SubstitutionPolicy(
            similarity,
            gate_threshold=config.gate_protect_threshold if config.substitution_enabled else 0.0,
            similarity_threshold=config.similarity_threshold,
        )
        self.eviction = self._build_eviction()
        self.admission = self._build_admission()

    def _build_eviction(self):
        if self.config.eviction == "lru":
            return LRUEviction()
        if self.config.eviction == "gate":
            return GateScoreEviction(self.history)
        if self.config.eviction == "coverage":
            return DiversityEviction(
                self.history,
                self.similarity,
                similarity_threshold=self.config.similarity_threshold,
                lam=self.config.coverage_lambda,
                k_min=self.config.coverage_k,
            )
        raise ValueError(self.config.eviction)

    def _build_admission(self):
        c = self.config
        if c.admission == "random":
            return BalancedRandomAdmission(c.seed)
        if c.admission == "greedy_current":
            return GreedyCurrentAdmission()
        if c.admission == "greedy_path":
            return GreedyPathAdmission(c.path_eta)
        if c.admission == "hungarian_current":
            return HungarianAdmission(name="hungarian_current")
        if c.admission == "hungarian_same":
            return HungarianAdmission(
                use_same=True,
                same_alpha=c.same_layer_alpha,
                name="hungarian_same",
            )
        if c.admission == "hungarian_same_path":
            return HungarianAdmission(
                use_same=True,
                use_path=True,
                same_alpha=c.same_layer_alpha,
                path_eta=c.path_eta,
                name="hungarian_same_path",
            )
        if c.admission == "hungarian_swap":
            seed = HungarianAdmission(
                use_same=True,
                use_path=True,
                same_alpha=c.same_layer_alpha,
                path_eta=c.path_eta,
                name="hungarian_same_path",
            )
            return SwapRefinedAdmission(seed)
        raise ValueError(c.admission)

    def plan_layer(self, routes: LayerRoutes) -> LayerPlan:
        if routes.full_router_probs is not None:
            self.history.update(routes.layer, routes.full_router_probs)

        decision = self.substitution.decide(routes, self.cache)
        effective = merge_effective_routes(routes, decision)
        execution_experts = {e for token in effective for e in token}

        # Owners that already exist before this layer's admissions.
        preowned = {
            e: self.cache.owner_of((routes.layer, e))
            for e in execution_experts
            if self.cache.resident((routes.layer, e))
        }
        preowned = {e: int(r) for e, r in preowned.items() if r is not None}

        incoming = sorted(decision.residual_exact_misses)
        adm_ctx = AdmissionContext(
            layer=routes.layer,
            incoming=incoming,
            origin_ranks=routes.origin_ranks,
            effective_token_routes=effective,
            preowned=preowned,
            cache=self.cache,
            world_size=self.config.world_size,
            affinity=self.affinity,
        )
        admission = self.admission.place(adm_ctx)

        local_exec = {
            r: LocalExecPlan() for r in range(self.config.world_size)
        }
        # Existing executing experts must not be evicted by same-event misses.
        pinned = {(routes.layer, e) for e in execution_experts if e in preowned}

        # Reject an impossible event before modifying logical residency. Hard
        # quotas and pinned execution experts must never cause partial plans.
        for rank, quota in enumerate(admission.quotas):
            available = self.cache.ranks[rank].capacity - sum(
                self.cache.owner_of(key) == rank for key in pinned
            )
            if quota > available:
                raise RuntimeError(
                    f"rank {rank}: admission quota {quota} exceeds {available} unpinned slots; "
                    "increase cache capacity or reduce the event token batch"
                )

        # Existing hits first.
        for e, rank in sorted(preowned.items()):
            slot = self.cache.ranks[rank].slot_of((routes.layer, e))
            local_exec[rank].hit_ops.append((routes.layer, e, slot))

        tick = self.cache.next_tick()
        order_by_rank = [0] * self.config.world_size

        # Admit exact misses in deterministic expert order. Newly admitted
        # execution experts become pinned immediately.
        for e in incoming:
            rank = admission.expert_to_rank[e]
            rank_cache = self.cache.ranks[rank]
            free = rank_cache.free_slot()
            victim_layer = -1
            victim_expert = -1
            if free is None:
                victim = self.eviction.choose(
                    self.cache, rank, routes.layer, pinned
                )
                victim_layer, victim_expert = victim
                _victim_rank, free = self.cache.evict(victim)
                if _victim_rank != rank:
                    raise AssertionError("victim rank mismatch")

            self.cache.place(rank, (routes.layer, e), free, tick)
            pinned.add((routes.layer, e))
            local_exec[rank].miss_ops.append(
                (
                    routes.layer,
                    e,
                    free,
                    victim_layer,
                    victim_expert,
                    order_by_rank[rank],
                )
            )
            order_by_rank[rank] += 1

        # All execution experts now have an owner.
        owners: Dict[int, int] = {}
        for e in execution_experts:
            owner = self.cache.owner_of((routes.layer, e))
            if owner is None:
                raise RuntimeError(f"execution expert {e} has no owner")
            owners[e] = int(owner)
            self.cache.touch((routes.layer, e), tick)

        self.cache.assert_consistent()

        return LayerPlan(
            layer=routes.layer,
            substitution=decision,
            admission=admission,
            owner_by_expert=owners,
            local_exec=local_exec,
            effective_token_routes=effective,
            pinned_keys=pinned,
        )
