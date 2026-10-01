"""Prefetch follows the active fetch policy.

After the 2026-05-26 single-controller refactor the actual production path
goes through ``CommandDispatcher.execute`` (controller/command_dispatcher.py),
which applies the same per-rank filter — ``if op.target_rank != rank:
continue`` — when issuing archer commands.  The legacy ``_issue_prefetch``
in ep_executor is gone.

These tests still guard the underlying invariant: changing the fetch policy
(naive vs balanced) deterministically changes per-rank work distribution,
no separate prefetch knob.  The policies/ module is kept as a parallel
implementation for tests; the controller's placement_planner uses
equivalent logic in production.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from moe_infinity_ep.policies.balanced_fetch import BalancedFetchPolicy
from moe_infinity_ep.policies.base import LaneLoads
from moe_infinity_ep.policies.naive_fetch import NaiveFetchPolicy


def _count_per_rank_prefetch(fetch_ops, ep_size: int) -> Dict[int, int]:
    """Mirror the filter logic in ``_issue_prefetch``: per-rank count of
    ops whose ``target_rank`` matches that rank."""
    counts = {r: 0 for r in range(ep_size)}
    for op in fetch_ops:
        counts[op.target_rank] += 1
    return counts


def test_naive_prefetch_distribution_matches_modulo():
    """Under naive, prefetch count per rank = miss count where expert_id % ep == rank."""
    ep_size = 2
    pol = NaiveFetchPolicy(ep_size=ep_size)
    # 100 misses with expert_ids alternating odd/even
    misses = [(0, e) for e in range(100)]
    ops = pol.decide(misses, LaneLoads.zeros(ep_size), cache_view=None)
    counts = _count_per_rank_prefetch(ops, ep_size)
    # 50 even (→ rank 0), 50 odd (→ rank 1)
    assert counts == {0: 50, 1: 50}


def test_balanced_prefetch_distribution_with_skewed_starting_load():
    """Under balanced with rank 0 already heavy, new misses skew to rank 1."""
    ep_size = 2
    pol = BalancedFetchPolicy(ep_size=ep_size, expert_bytes=10_000_000)
    # rank 0 already has 50 experts worth of load = 500MB
    skewed = LaneLoads(
        pcie_bytes_outstanding=[50 * 10_000_000, 0],
        nvlink_bytes_outstanding=[0, 0],
    )
    misses = [(0, e) for e in range(50)]
    ops = pol.decide(misses, skewed, cache_view=None)
    counts = _count_per_rank_prefetch(ops, ep_size)
    # All 50 new misses should land on rank 1 until it catches up to rank 0.
    assert counts[1] == 50
    assert counts[0] == 0


def test_balanced_prefetch_equal_starting_load_alternates():
    """Equal starting loads + balanced + deterministic tie-break (lower rank
    index wins ties) → first miss to rank 0, then rank 1, alternating."""
    ep_size = 2
    pol = BalancedFetchPolicy(ep_size=ep_size, expert_bytes=10_000_000)
    misses = [(0, e) for e in range(10)]
    ops = pol.decide(misses, LaneLoads.zeros(ep_size), cache_view=None)
    counts = _count_per_rank_prefetch(ops, ep_size)
    # Perfect 5/5 split.
    assert counts == {0: 5, 1: 5}
    # And the SEQUENCE alternates (op[0] → rank 0, op[1] → rank 1, ...).
    expected_seq = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    assert [op.target_rank for op in ops] == expected_seq


def test_prefetch_followthrough_invariant_under_4_ranks():
    """4-rank scenario: balanced distributes deterministically by min load."""
    ep_size = 4
    pol = BalancedFetchPolicy(ep_size=ep_size, expert_bytes=10_000_000)
    # 12 misses, equal initial loads → each rank gets 3.
    misses = [(0, e) for e in range(12)]
    ops = pol.decide(misses, LaneLoads.zeros(ep_size), cache_view=None)
    counts = _count_per_rank_prefetch(ops, ep_size)
    assert counts == {0: 3, 1: 3, 2: 3, 3: 3}


def test_prefetch_followthrough_skips_other_ranks():
    """The executor's `target_rank == my_rank` filter must mean rank 0's
    prefetch only fires for ops assigned to rank 0."""
    ep_size = 2
    pol = NaiveFetchPolicy(ep_size=ep_size)
    misses = [(0, e) for e in range(10)]
    ops = pol.decide(misses, LaneLoads.zeros(ep_size), cache_view=None)
    # Simulate _issue_prefetch on rank 0:
    my_rank = 0
    rank0_targets = [op for op in ops if op.target_rank == my_rank]
    # Even expert ids only.
    assert all(op.expert_id % 2 == 0 for op in rank0_targets)
    # Simulate rank 1:
    rank1_targets = [op for op in ops if op.target_rank == 1]
    assert all(op.expert_id % 2 == 1 for op in rank1_targets)
    # Together = full set, no overlap.
    assert len(rank0_targets) + len(rank1_targets) == len(ops)
