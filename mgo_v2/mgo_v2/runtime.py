from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from .communicator import dispatch_tokens, return_partials
from .controller import GlobalExpertController
from .executor import LegacySlotExecutorAdapter
from .types import LayerRoutes


@dataclass
class GatheredRoutes:
    routes: LayerRoutes
    counts: list[int]
    local_offset: int


def _all_gather_padded(tensor: torch.Tensor, counts: list[int]) -> list[torch.Tensor]:
    world = dist.get_world_size()
    max_n = max(counts) if counts else 0
    padded_shape = (max_n,) + tuple(tensor.shape[1:])
    padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
    if tensor.shape[0]:
        padded[: tensor.shape[0]] = tensor
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    return [g[: counts[r]] for r, g in enumerate(gathered)]


def gather_global_routes(
    layer: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    full_router_probs: torch.Tensor,
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
    dist.all_gather(count_buf, local_count)
    counts = [int(x.item()) for x in count_buf]

    all_selected = _all_gather_padded(selected_experts.to(torch.int64), counts)
    all_weights = _all_gather_padded(routing_weights, counts)
    all_probs = _all_gather_padded(full_router_probs.to(torch.float32), counts)

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
    ):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        if dist.get_world_size() != controller.config.world_size:
            raise RuntimeError("world size != controller config")
        self.controller = controller
        self.executor = executor

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
            layer, selected_experts, routing_weights, full_router_probs
        )
        # Every rank has the same metadata and deterministic controller state,
        # so every rank independently derives the same plan.
        plan = self.controller.plan_layer(gathered.routes)

        n_local = hidden_states.shape[0]
        start = gathered.local_offset
        local_routes = plan.effective_token_routes[start : start + n_local]

        received = dispatch_tokens(
            hidden_states,
            local_routes,
            plan.owner_by_expert,
            self.controller.config.top_k,
        )
        local_plan = plan.local_exec[dist.get_rank()]
        partial = self.executor.execute(
            layer, received, local_plan, is_decode=is_decode
        )
        return return_partials(partial, received, n_local)
