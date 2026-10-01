"""AffinityMigrationPlacement — unit tests with synthetic ClassifyResult."""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from moe_infinity_ep.cache.view import ClassifyResult, GlobalCacheView
from moe_infinity_ep.policies.affinity_migration import (
    AffinityMigrationPlacement,
)
from moe_infinity_ep.policies.base import FetchOp


def _mk_view(ep_size: int = 2, capacity: int = 8, num_experts: int = 4):
    # ep_group=None — apply_placements doesn't issue collectives.
    return GlobalCacheView(
        ep_size=ep_size,
        capacity_per_rank=capacity,
        num_layers=2,
        num_experts=num_experts,
        ep_group=None,
    )


def _classify(hits: Dict, ep_size: int, num_experts: int,
              per_rank: List[List[int]]) -> ClassifyResult:
    return ClassifyResult(
        hits=hits,
        misses=[],
        demand_vector=torch.tensor([1] * num_experts, dtype=torch.int8),
        per_rank_demand=torch.tensor(per_rank, dtype=torch.int8),
    )


def test_no_migration_when_demand_balanced():
    """Both ranks demand expert 0 equally → policy should leave it put."""
    view = _mk_view(ep_size=2, num_experts=4)
    view.install(rank=0, layer_id=0, expert_id=0)
    pol = AffinityMigrationPlacement(ep_size=2, max_migrations_per_layer=1,
                                     margin=1.5)
    classify = _classify(
        hits={(0, 0): 0},
        ep_size=2, num_experts=4,
        per_rank=[[1, 0, 0, 0], [1, 0, 0, 0]],
    )
    ops = pol.decide(classify, fetch_ops=[], cache_view=view)
    assert ops == []


def test_migration_when_other_rank_demand_dominant():
    """Expert 0 is on rank 0, but only rank 1 demands it — migrate."""
    view = _mk_view(ep_size=2, num_experts=4)
    view.install(rank=0, layer_id=0, expert_id=0)
    pol = AffinityMigrationPlacement(ep_size=2, max_migrations_per_layer=1,
                                     margin=1.5)
    classify = _classify(
        hits={(0, 0): 0},
        ep_size=2, num_experts=4,
        per_rank=[[0, 0, 0, 0], [1, 0, 0, 0]],
    )
    ops = pol.decide(classify, fetch_ops=[], cache_view=view)
    assert len(ops) == 1
    op = ops[0]
    assert op.layer_id == 0 and op.expert_id == 0
    assert op.source == "nvlink"
    assert op.source_rank == 0
    assert op.target_rank == 1
    assert op.evict_key is None  # rank 1 had free slots


def test_apply_placements_migrates_locate_index():
    """End-to-end: apply_placements actually moves the expert between ranks."""
    view = _mk_view(ep_size=2, num_experts=4)
    view.install(rank=0, layer_id=0, expert_id=0)
    assert view.locate(0, 0) == 0   # initially on rank 0

    pol = AffinityMigrationPlacement(ep_size=2)
    classify = _classify(
        hits={(0, 0): 0},
        ep_size=2, num_experts=4,
        per_rank=[[0, 0, 0, 0], [1, 0, 0, 0]],
    )
    ops = pol.decide(classify, fetch_ops=[], cache_view=view)
    view.apply_placements(ops)
    assert view.locate(0, 0) == 1   # migrated to rank 1
    assert (0, 0) not in view.cache[0]
    assert (0, 0) in view.cache[1]


def test_migration_budget_caps_count():
    """Two candidate migrations exist; budget=1 picks only one."""
    view = _mk_view(ep_size=2, num_experts=4)
    view.install(rank=0, layer_id=0, expert_id=0)
    view.install(rank=0, layer_id=0, expert_id=1)
    pol = AffinityMigrationPlacement(ep_size=2, max_migrations_per_layer=1,
                                     margin=1.5)
    classify = _classify(
        hits={(0, 0): 0, (0, 1): 0},
        ep_size=2, num_experts=4,
        per_rank=[[0, 0, 0, 0], [1, 1, 0, 0]],
    )
    ops = pol.decide(classify, fetch_ops=[], cache_view=view)
    # Budget=1, deterministic by (layer, expert) sort, so (0,0) migrates first.
    assert len(ops) == 1
    assert (ops[0].layer_id, ops[0].expert_id) == (0, 0)


def test_migration_combined_with_miss_placements():
    """Miss placement runs first; migration after — both end up in op list."""
    view = _mk_view(ep_size=2, num_experts=4, capacity=8)
    view.install(rank=0, layer_id=0, expert_id=0)
    pol = AffinityMigrationPlacement(ep_size=2, max_migrations_per_layer=1)
    classify = _classify(
        hits={(0, 0): 0},
        ep_size=2, num_experts=4,
        per_rank=[[0, 0, 0, 0], [1, 0, 0, 0]],
    )
    fetch_ops = [FetchOp(layer_id=0, expert_id=2, target_rank=0)]
    ops = pol.decide(classify, fetch_ops=fetch_ops, cache_view=view)
    # 1 miss placement + 1 migration
    assert len(ops) == 2
    miss_op = next(o for o in ops if o.source == "pcie")
    mig_op = next(o for o in ops if o.source == "nvlink")
    assert miss_op.expert_id == 2 and miss_op.target_rank == 0
    assert mig_op.expert_id == 0 and mig_op.target_rank == 1


def test_migration_skipped_when_target_full_and_only_victim_is_self():
    """Target rank's only LRU victim is the expert being migrated → skip."""
    view = _mk_view(ep_size=2, num_experts=2, capacity=1)
    # rank 1 already holds (0, 0); rank 0 also has it (locate returns min).
    view.install(rank=0, layer_id=0, expert_id=0)
    view.install(rank=1, layer_id=0, expert_id=0)
    pol = AffinityMigrationPlacement(ep_size=2)
    classify = _classify(
        hits={(0, 0): 0},
        ep_size=2, num_experts=2,
        per_rank=[[0, 0], [1, 0]],
    )
    ops = pol.decide(classify, fetch_ops=[], cache_view=view)
    # rank 1 is full (capacity=1) and its only LRU candidate IS (0,0)
    # which is what we'd be installing — skip migration.
    assert ops == []
