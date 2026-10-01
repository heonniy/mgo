"""M5 smoke tests: registry + ABC + plug-in third-party policy.

These tests are pure-Python (no CUDA, no NCCL) and exercise:
1. All registered fetch/placement policies instantiate via the registry.
2. ABC contract: third-party class that satisfies the interface is accepted;
   one that doesn't is rejected.
3. Smoke policy (``smoke_rr_layer`` + ``smoke_passthrough``) drives a synthetic
   layer end-to-end through the same pure functions the executor uses, with
   correct fetch counts, valid target_rank, and one PlacementOp per FetchOp.
"""
from __future__ import annotations

import pytest

from moe_infinity_ep.cache.view import GlobalCacheView
from moe_infinity_ep.policies.base import (
    FetchDispatchPolicy, FetchOp, LaneLoads,
    PlacementOp, PlacementPolicy,
)
from moe_infinity_ep.policies.registry import (
    list_fetch_policies, list_placement_policies,
    load_fetch_policy, load_placement_policy,
    register_fetch_policy, register_placement_policy,
)
from moe_infinity_ep.cache.view import ClassifyResult


EP_SIZE = 2
NUM_EXPERTS = 8
NUM_LAYERS = 4


def _make_view() -> GlobalCacheView:
    return GlobalCacheView(
        ep_size=EP_SIZE,
        capacity_per_rank=4,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        ep_group=None,
    )


def _zero_loads() -> LaneLoads:
    return LaneLoads.zeros(EP_SIZE)


def _classify(hits=None, misses=None) -> ClassifyResult:
    return ClassifyResult(
        hits=hits or {}, misses=misses or [],
        demand_vector=None, per_rank_demand=None,
    )


# ---------- 1. Registry can list and load every registered policy ----------

def test_registry_lists_known_policies():
    fetch = list_fetch_policies()
    placement = list_placement_policies()
    assert "naive" in fetch and "balanced" in fetch and "smoke_rr_layer" in fetch
    assert "naive_token_route" in placement and "smoke_passthrough" in placement


@pytest.mark.parametrize("name", ["naive", "balanced", "smoke_rr_layer"])
def test_load_fetch_policy_returns_abc(name):
    pol = load_fetch_policy(name, ep_size=EP_SIZE)
    assert isinstance(pol, FetchDispatchPolicy)


@pytest.mark.parametrize(
    "name", ["naive_token_route", "naive", "smoke_passthrough"],
)
def test_load_placement_policy_returns_abc(name):
    pol = load_placement_policy(name, ep_size=EP_SIZE)
    assert isinstance(pol, PlacementPolicy)


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        load_fetch_policy("no_such_policy", ep_size=EP_SIZE)
    with pytest.raises(ValueError):
        load_placement_policy("also_missing", ep_size=EP_SIZE)


# ---------- 2. ABC compliance for runtime-registered third-party classes ----------

class _ExternalFetch(FetchDispatchPolicy):
    """Third-party policy: all misses go to rank 0."""
    def __init__(self, ep_size):
        self.ep_size = ep_size

    def decide(self, misses, lane_loads, cache_view):
        return [
            FetchOp(layer_id=l, expert_id=e, target_rank=0, lane=0)
            for (l, e) in misses
        ]


class _NotAPolicy:
    pass


def test_register_external_fetch_policy():
    register_fetch_policy("external", _ExternalFetch)
    pol = load_fetch_policy("external", ep_size=EP_SIZE)
    out = pol.decide([(0, 1), (1, 3)], _zero_loads(), _make_view())
    assert all(op.target_rank == 0 for op in out)
    assert [op.expert_id for op in out] == [1, 3]


def test_register_non_abc_class_rejected():
    with pytest.raises(TypeError):
        register_fetch_policy("bad", _NotAPolicy)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        register_placement_policy("bad", _NotAPolicy)  # type: ignore[arg-type]


# ---------- 3. Smoke policy end-to-end on synthetic inputs ----------

def test_smoke_fetch_distributes_by_layer():
    pol = load_fetch_policy("smoke_rr_layer", ep_size=EP_SIZE)
    misses = [(layer, e) for layer in range(NUM_LAYERS) for e in (0, 1, 2)]
    ops = pol.decide(misses, _zero_loads(), _make_view())
    assert len(ops) == len(misses)
    for op in ops:
        assert 0 <= op.target_rank < EP_SIZE
        # Smoke contract: target_rank = layer_id % ep_size.
        assert op.target_rank == op.layer_id % EP_SIZE


def test_smoke_placement_one_op_per_fetch_and_valid_eviction():
    view = _make_view()
    # Pre-fill rank 0's cache to its capacity (=4) so we exercise eviction.
    for e in range(4):
        view.install(0, layer_id=0, expert_id=e)
    fetch_pol = load_fetch_policy("smoke_rr_layer", ep_size=EP_SIZE)
    place_pol = load_placement_policy("smoke_passthrough", ep_size=EP_SIZE)

    # Layer 0 misses (will all target rank 0 under smoke_rr_layer).
    misses = [(0, 4), (0, 5)]
    fetch_ops = fetch_pol.decide(misses, _zero_loads(), view)
    assert all(op.target_rank == 0 for op in fetch_ops)

    plan = place_pol.decide(
        _classify(misses=misses), fetch_ops, view,
    )
    assert len(plan) == len(fetch_ops)
    for placement_op, fetch_op in zip(plan, fetch_ops):
        assert isinstance(placement_op, PlacementOp)
        assert placement_op.target_rank == fetch_op.target_rank
        assert placement_op.layer_id == fetch_op.layer_id
        assert placement_op.expert_id == fetch_op.expert_id
        # Capacity was full → expect an eviction victim (some (l,e) key).
        assert placement_op.evict_key is not None


def test_smoke_policy_end_to_end_apply():
    """Plan + apply on a real GlobalCacheView; verify state stays consistent."""
    view = _make_view()
    fetch_pol = load_fetch_policy("smoke_rr_layer", ep_size=EP_SIZE)
    place_pol = load_placement_policy("smoke_passthrough", ep_size=EP_SIZE)

    misses = [(0, 1), (1, 2), (1, 4)]  # mix of layers → targets via layer % 2
    fetch_ops = fetch_pol.decide(misses, _zero_loads(), view)
    plan = place_pol.decide(_classify(misses=misses), fetch_ops, view)
    view.apply_placements(plan)

    # Every installed key is now locatable.
    for op in plan:
        ranks = view.locate_index.get((op.layer_id, op.expert_id))
        assert ranks is not None and op.target_rank in ranks
    # Total cache entries == number of installs (no extra growth or shrink).
    total = sum(len(view.cache[r]) for r in range(EP_SIZE))
    assert total == len(plan)


# ---------- 4. ep_size parametrization (8-GPU forward compatibility) ----------

@pytest.mark.parametrize("ep_size", [2, 4, 8])
@pytest.mark.parametrize("policy_name", ["naive", "balanced", "smoke_rr_layer"])
def test_fetch_policy_target_rank_in_range(ep_size, policy_name):
    """Every FetchOp's target_rank must be within [0, ep_size)."""
    pol = load_fetch_policy(policy_name, ep_size=ep_size)
    view = GlobalCacheView(
        ep_size=ep_size, capacity_per_rank=4,
        num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS, ep_group=None,
    )
    misses = [(l, e) for l in range(NUM_LAYERS) for e in range(NUM_EXPERTS)]
    ops = pol.decide(misses, LaneLoads.zeros(ep_size), view)
    for op in ops:
        assert 0 <= op.target_rank < ep_size


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_smoke_passthrough_budgeting_at_scale(ep_size):
    """Each fetch landing on the same rank gets a distinct evict_key."""
    pol = load_placement_policy("smoke_passthrough", ep_size=ep_size)
    view = GlobalCacheView(
        ep_size=ep_size, capacity_per_rank=2,
        num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS, ep_group=None,
    )
    # Pre-fill rank 0 to capacity.
    for e in range(2):
        view.install(0, layer_id=0, expert_id=e)
    # 4 fetches all targeting rank 0 — must produce 2 distinct evict_keys
    # (cap=2) and then no eviction (no free room left, no more LRU candidates).
    fetch_ops = [
        FetchOp(layer_id=1, expert_id=e, target_rank=0, lane=0)
        for e in range(4)
    ]
    plan = pol.decide(_classify(), fetch_ops, view)
    evict_keys = [op.evict_key for op in plan if op.evict_key is not None]
    assert len(evict_keys) == len(set(evict_keys)), \
        f"duplicate evict_keys: {evict_keys}"
