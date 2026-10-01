"""DemandCollector — per-layer NCCL all_gather of expert demand.

Wraps the previously-loose ``cache.sync.sync_and_classify`` flow into a
controller-owned component. Output is byte-identical on every rank →
GlobalCacheView state stays in lockstep across ranks.

Two entry points:
  * ``collect``           — blocking. Returns the union of demanded experts.
  * ``start`` / ``finish`` — split for pipeline overlap (M7 path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist


@dataclass
class AsyncHandle:
    work: object               # torch.distributed.Work
    per_rank: torch.Tensor     # [ep_size, num_experts] int8
    layer_id: int


class DemandCollector:
    """NCCL all_gather of per-rank demand → union of demanded experts.

    The collector does NOT classify hits/misses; that's the controller's
    job (it owns the cache_view). The collector only produces the union
    of demanded ``expert_id``s as a sorted list of ints.
    """

    def __init__(self, ep_size: int, num_experts: int, ep_group):
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.ep_group = ep_group

    # ---- blocking path ----

    def collect(self, layer_id: int, local_router_mask: torch.Tensor) -> List[int]:
        """Synchronous: all_gather + return demanded expert_ids (sorted)."""
        per_rank = self._gather(local_router_mask)
        return self._extract_demanded(per_rank)

    # ---- new architecture: per-rank token count ----

    def collect_with_count(
        self, layer_id: int, local_router_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cooperative-offloading용 collect — token COUNT 보존.

        Returns int64 tensor of shape ``[ep_size, num_experts]`` on CPU:
        ``per_rank[r, e]`` = rank r 의 local token 중 expert e 를 demand
        하는 token 수.

        Controller 가 이걸로:
          * union demand:   ``per_rank.any(dim=0).nonzero()``
          * demand count:   ``per_rank.sum(dim=0)``
          * per-rank set:   ``per_rank[r].nonzero()``
        세 가지 모두 한 collective 결과로 derive.
        """
        device = local_router_mask.device
        # router_mask: [N_local, num_experts] bool
        local_count = (
            local_router_mask.view(-1, self.num_experts)
            .sum(dim=0).to(torch.int64).contiguous()
        )
        per_rank = torch.empty(
            self.ep_size, self.num_experts,
            dtype=torch.int64, device=device,
        )
        dist.all_gather(
            list(per_rank.unbind(dim=0)),
            local_count,
            group=self.ep_group,
        )
        return per_rank.cpu()

    # ---- pipeline overlap path ----

    def start(self, layer_id: int, local_router_mask: torch.Tensor) -> AsyncHandle:
        device = local_router_mask.device
        local_demand = (
            local_router_mask.view(-1, self.num_experts).sum(dim=0) > 0
        ).to(torch.int8).contiguous().view(self.num_experts)
        per_rank = torch.empty(
            self.ep_size, self.num_experts, dtype=torch.int8, device=device,
        )
        gather_list = list(per_rank.unbind(dim=0))
        work = dist.all_gather(
            gather_list, local_demand, group=self.ep_group, async_op=True,
        )
        return AsyncHandle(work=work, per_rank=per_rank, layer_id=layer_id)

    def finish(self, handle: AsyncHandle) -> List[int]:
        handle.work.wait()
        return self._extract_demanded(handle.per_rank)

    # ---- internals ----

    def _gather(self, local_router_mask: torch.Tensor) -> torch.Tensor:
        device = local_router_mask.device
        local_demand = (
            local_router_mask.view(-1, self.num_experts).sum(dim=0) > 0
        ).to(torch.int8).contiguous().view(self.num_experts)
        per_rank = torch.empty(
            self.ep_size, self.num_experts, dtype=torch.int8, device=device,
        )
        gather_list = list(per_rank.unbind(dim=0))
        dist.all_gather(gather_list, local_demand, group=self.ep_group)
        return per_rank

    def _extract_demanded(self, per_rank: torch.Tensor) -> List[int]:
        union = per_rank.any(dim=0).to(torch.int8)
        return union.cpu().nonzero(as_tuple=True)[0].tolist()
