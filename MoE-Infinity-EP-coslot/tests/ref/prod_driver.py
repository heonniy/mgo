"""Adapter that drives the PRODUCTION GlobalCacheController and returns output
in the same normalized shape as the reference oracle's ``RefController.step``.

This is the only file in tests/ref that imports moe_infinity_ep — it is the
bridge for differential testing.  reference.py itself stays import-pure.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from moe_infinity_ep.controller.global_controller import GlobalCacheController
from moe_infinity_ep.controller.owner_policy import build_owner_policy
from moe_infinity_ep.controller.evict_policy import build_evict_policy
from moe_infinity_ep.controller.rank_planner import RankPlanner
from moe_infinity_ep.controller.ours_planner import OursPlacementPlanner

NUM_LAYERS = 64


class SynthCollector:
    """Drop-in DemandCollector returning a preset [ep_size, num_experts] tensor.

    Only used for ep_size > 1 (for ep_size == 1 the controller takes a stub
    path off the router mask — see prod_step).
    """

    def __init__(self, ep_size: int, num_experts: int):
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.ep_group = None
        self._cur: Optional[torch.Tensor] = None

    def set_matrix(self, matrix: List[List[int]]) -> None:
        self._cur = torch.tensor(matrix, dtype=torch.int64)

    def collect_with_count(self, layer_id, local_router_mask):
        assert self._cur is not None
        return self._cur


def make_prod(ep_size: int, cap: int, num_experts: int,
              owner: str, evict: str) -> GlobalCacheController:
    return GlobalCacheController(
        ep_size=ep_size, ep_rank=0, cap_per_rank=cap,
        num_layers=NUM_LAYERS, num_experts=num_experts, ep_group=None,
        owner_policy=build_owner_policy(owner),
        rank_planner=RankPlanner(build_evict_policy(evict)),
        demand_collector=SynthCollector(ep_size, num_experts),
        command_dispatcher=None,
    )


def make_prod_ours(ep_size: int, cap: int, num_experts: int,
                   future_lambda: float = 0.0,
                   evict_cost_provider=None,
                   future_affinity_provider=None) -> GlobalCacheController:
    """Production controller in OURS joint-placement mode (baseline #4)."""
    planner = OursPlacementPlanner(
        ep_size=ep_size, future_lambda=future_lambda,
        evict_cost_provider=evict_cost_provider,
        future_affinity_provider=future_affinity_provider)
    return GlobalCacheController(
        ep_size=ep_size, ep_rank=0, cap_per_rank=cap,
        num_layers=NUM_LAYERS, num_experts=num_experts, ep_group=None,
        owner_policy=build_owner_policy("static_placement"),   # unused placeholder
        rank_planner=RankPlanner(build_evict_policy("lfu")),    # unused placeholder
        demand_collector=SynthCollector(ep_size, num_experts),
        ours_planner=planner)


def _mask_from_row(row: List[int], num_experts: int) -> torch.Tensor:
    """Build a [maxcount, num_experts] bool mask whose column sums == row.

    The ep_size==1 controller path derives demand by summing the router mask
    rows, so we encode the per-expert counts as top-aligned True columns.
    """
    maxc = max(row) if row and max(row) > 0 else 1
    mask = torch.zeros(maxc, num_experts, dtype=torch.bool)
    for e, c in enumerate(row):
        if c > 0:
            mask[:c, e] = True
    return mask


def seed_prod(c: GlobalCacheController, shadow: Dict[int, List]) -> None:
    for r, slots in shadow.items():
        rc = c.cache_view.per_rank[int(r)]
        for i, cell in enumerate(slots):
            if cell is not None:
                rc.apply(slot=i, evict=None,
                         insert=(int(cell[0]), int(cell[1])), demand_count=0)


def prod_step(c: GlobalCacheController, matrix: List[List[int]],
              layer_id: int = 0, priority=None, is_decode: bool = False) -> dict:
    ne = c.num_experts
    if c.ep_size == 1:
        mask = _mask_from_row(matrix[0], ne)
    else:
        c.demand_collector.set_matrix(matrix)
        mask = torch.zeros(1, ne, dtype=torch.bool)

    p1 = c.classify_and_kick_hit(layer_id, mask)
    plan = c.plan_misses(p1, layer_id, priority=priority, is_decode=is_decode)

    def ek(k):
        return None if k is None else (int(k[0]), int(k[1]))

    hit_ops = [{"expert": ek(op.expert), "owner_rank": op.owner_rank,
                "slot": op.slot} for op in p1.hit_launch_info.hit_ops]
    miss_per_rank = {r: [ek(k) for k in p1.miss_per_rank.get(r, [])]
                     for r in range(c.ep_size)}
    # ``miss_set`` is the load-bearing union set (always populated); for baselines
    # it equals union(miss_per_rank) and for OURS miss_per_rank is empty.
    misses = sorted(ek(k) for k in p1.miss_set)
    fetch_ops = [{"expert": ek(op.expert), "fetcher_rank": op.fetcher_rank,
                  "dst_slot": op.dst_slot, "victim_expert": ek(op.victim_expert),
                  "order": op.order} for op in plan.fetch_ops]
    # fetch grouped by the rank that actually fetches (robust to ours/baseline).
    fetch_per_rank = {r: sorted(op["expert"] for op in fetch_ops
                                if op["fetcher_rank"] == r)
                      for r in range(c.ep_size)}
    routing_map = {ek(k): v for k, v in plan.expert_to_rank.items()}
    demand = {ek(k): int(v) for k, v in p1.demand.items()}
    shadow = {r: [ek(s) for s in c.cache_view.per_rank[r].slots]
              for r in range(c.ep_size)}
    return {"demand": demand, "hit_ops": hit_ops,
            "miss_per_rank": miss_per_rank, "misses": misses,
            "fetch_ops": fetch_ops, "fetch_per_rank": fetch_per_rank,
            "routing_map": routing_map, "shadow_after": shadow}


def prod_bump(c: GlobalCacheController) -> None:
    # command_dispatcher is None -> verify_layer_end just bumps layer_seq.
    c.verify_layer_end(0)


def prod_plan_tuples(c: GlobalCacheController, matrix: List[List[int]],
                     layer_id: int = 0, is_decode: bool = False):
    """Run one layer and return per-rank archer-submit tuples + shadow.

    Returns (hits_by_rank, misses_by_rank, shadow_after) where
      hits_by_rank[r]  = [(layer, expert, slot), ...]
      misses_by_rank[r]= [(layer, expert, dst_slot, victim_l, victim_e, order)]
    matching the C++ ExpertDispatcher.submit_plan signature.  Mutates the
    controller shadow (Phase 2), like the real executor.
    """
    ne = c.num_experts
    if c.ep_size == 1:
        mask = _mask_from_row(matrix[0], ne)
    else:
        c.demand_collector.set_matrix(matrix)
        mask = torch.zeros(1, ne, dtype=torch.bool)

    p1 = c.classify_and_kick_hit(layer_id, mask)
    plan = c.plan_misses(p1, layer_id, is_decode=is_decode)

    hits_by_rank = {r: [] for r in range(c.ep_size)}
    for op in p1.hit_launch_info.hit_ops:
        hits_by_rank[op.owner_rank].append(
            (int(op.expert[0]), int(op.expert[1]), int(op.slot)))

    misses_by_rank = {r: [] for r in range(c.ep_size)}
    for op in plan.fetch_ops:
        if op.victim_expert is None:
            vl, ve = -1, -1
        else:
            vl, ve = int(op.victim_expert[0]), int(op.victim_expert[1])
        misses_by_rank[op.fetcher_rank].append(
            (int(op.expert[0]), int(op.expert[1]), int(op.dst_slot),
             vl, ve, int(op.order)))

    shadow = {r: [(int(s[0]), int(s[1])) if s is not None else None
                  for s in c.cache_view.per_rank[r].slots]
              for r in range(c.ep_size)}
    return hits_by_rank, misses_by_rank, shadow
