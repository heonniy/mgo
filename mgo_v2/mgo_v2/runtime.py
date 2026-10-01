from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.distributed as dist

from .communicator import dispatch_tokens, return_partials
from .controller import GlobalExpertController
from .executor import LegacySlotExecutorAdapter
from .types import LayerRoutes
from .metrics import ExpertMetrics


@dataclass
class GatheredRoutes:
    routes: LayerRoutes
    counts: list[int]
    local_offset: int


def _all_gather_padded(tensor: torch.Tensor, counts: list[int], stats=None) -> list[torch.Tensor]:
    world = dist.get_world_size()
    max_n = max(counts) if counts else 0
    padded_shape = (max_n,) + tuple(tensor.shape[1:])
    padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
    if tensor.shape[0]:
        padded[: tensor.shape[0]] = tensor
    gathered = [torch.empty_like(padded) for _ in range(world)]
    operation = lambda: dist.all_gather(gathered, padded)
    if stats is None:
        operation()
    else:
        stats.call("router_metadata", padded.numel() * padded.element_size() * (world - 1), operation)
    return [g[: counts[r]] for r, g in enumerate(gathered)]


def gather_global_routes(
    layer: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    full_router_probs: torch.Tensor,
    stats=None,
) -> GatheredRoutes:
    """Gather compact routing metadata without RPC or CPU tensor payloads.

    Tensors travel over the process group first; only controller metadata is
    copied to CPU after the collective.
    """
    device = selected_experts.device
    world = dist.get_world_size()
    rank = dist.get_rank()

    local_count = torch.tensor([selected_experts.shape[0]], dtype=torch.int64, device=device)
    count_buf = [torch.empty_like(local_count) for _ in range(world)]
    operation = lambda: dist.all_gather(count_buf, local_count)
    if stats is None:
        operation()
    else:
        stats.call("counts", local_count.element_size() * (world - 1), operation)
    counts = torch.cat(count_buf).tolist()

    all_selected = _all_gather_padded(selected_experts.to(torch.int64), counts, stats)
    all_weights = _all_gather_padded(routing_weights, counts, stats)
    all_probs = _all_gather_padded(full_router_probs.to(torch.float32), counts, stats)

    selected = torch.cat(all_selected, dim=0).cpu().numpy()
    weights = torch.cat(all_weights, dim=0).float().cpu().numpy()
    probs = torch.cat(all_probs, dim=0).float().cpu().numpy()
    origins = np.concatenate(
        [np.full(n, r, dtype=np.int64) for r, n in enumerate(counts)], axis=0
    )
    offset = sum(counts[:rank])
    return GatheredRoutes(
        routes=LayerRoutes(
            layer=layer,
            origin_ranks=origins,
            selected_experts=selected,
            routing_weights=weights,
            full_router_probs=probs,
        ),
        counts=counts,
        local_offset=offset,
    )


class DistributedMoERuntime:
    def __init__(
        self,
        controller: GlobalExpertController,
        executor: LegacySlotExecutorAdapter,
        debug: bool = False,
        observer=None,
        collective_stats=None,
    ):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        if dist.get_world_size() != controller.config.world_size:
            raise RuntimeError("world size != controller config")
        self.controller = controller
        self.executor = executor
        self.debug = debug
        self.observer = observer
        self.collective_stats = collective_stats
        self.metrics = ExpertMetrics()
        self.controller_seconds = 0.0
        self.events = 0
        self._fetched_keys = set()

    def forward_layer(
        self,
        layer: int,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        full_router_probs: torch.Tensor,
        is_decode: bool,
    ) -> torch.Tensor:
        gathered = gather_global_routes(
            layer, selected_experts, routing_weights, full_router_probs, self.collective_stats
        )
        # Every rank has the same metadata and deterministic controller state,
        # so every rank independently derives the same plan.
        start_time = time.perf_counter()
        plan = self.controller.plan_layer(gathered.routes)
        self.controller_seconds += time.perf_counter() - start_time
        self.metrics.update_substitution(plan.substitution)
        self.events += 1
        self.metrics.events += 1
        weights = gathered.routes.routing_weights
        mapped = np.isin(gathered.routes.selected_experts, list(plan.substitution.source_to_target))
        self.metrics.substituted_gate_mass += float(weights[mapped].sum(dtype=np.float64))
        self.metrics.total_gate_mass += float(weights.sum(dtype=np.float64))
        loads = np.zeros(self.controller.config.world_size, dtype=np.int64)
        for origin, routes in zip(gathered.routes.origin_ranks, plan.effective_token_routes):
            route_destinations = [plan.owner_by_expert[e] for e in routes]
            local_routes = route_destinations.count(int(origin))
            self.metrics.local_expert_routes += local_routes
            self.metrics.remote_expert_routes += len(route_destinations) - local_routes
            destinations = set(route_destinations)
            for dst in destinations:
                loads[dst] += 1
                self.metrics.remote_token_rank_pairs += dst != int(origin)
                self.metrics.local_token_rank_pairs += dst == int(origin)
        if loads.sum():
            self.metrics.rank_token_cv_sum += float(loads.std() / loads.mean())
            self.metrics.rank_token_max_mean_sum += float(loads.max() / loads.mean())
        incoming = {(layer, e) for e in plan.substitution.residual_exact_misses}
        self.metrics.reloads += len(incoming & self._fetched_keys)
        self._fetched_keys.update(incoming)

        n_local = hidden_states.shape[0]
        start = gathered.local_offset
        local_routes = plan.effective_token_routes[start : start + n_local]

        received = dispatch_tokens(
            hidden_states,
            local_routes,
            plan.owner_by_expert,
            self.controller.config.top_k,
            self.collective_stats,
        )
        local_plan = plan.local_exec[dist.get_rank()]
        partial = self.executor.execute(
            layer, received, local_plan, is_decode=is_decode
        )
        if self.debug:
            self.executor.assert_cache_matches(self.controller.cache.ranks[dist.get_rank()])
        output = return_partials(partial, received, n_local, self.collective_stats)
        if self.observer is not None:
            self.observer(gathered.routes, plan, output)
        return output
