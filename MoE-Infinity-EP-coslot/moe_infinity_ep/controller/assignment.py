"""Pure assignment / quota solvers for the OURS joint placement planner.

All functions here are **pure** (numpy / scipy only, no controller state) so they
can be unit-tested against brute-force optima in isolation.

Ported from
``/home/work/hyewon.lee/실험/prefill 알고리즘 실험/experiments/prefill_assignment/
prefill_assignment_solvers.py`` (``solve_hungarian`` / ``compute_k`` /
``evaluate_assignment`` / ``validate_assignment``) plus two new decode-side
solvers required by algorithm.md §2-3:

  * ``decode_quota_dp``  — prefix-constrained DP that chooses, per GPU, how many
    cache slots to open (q_g) so that ``sum q_g == M`` and the total prefix
    eviction-cost is minimised.
  * ``decode_match``     — Hungarian matching of miss experts to the selected
    physical slots, maximising placement benefit.

Determinism note: every solver here is deterministic for identical inputs.  The
controller relies on all EP ranks feeding **byte-identical** inputs (the demand
matrix is all-gathered; the cache view is replicated), so identical outputs across
ranks follow.  See ``decode_quota_dp`` for the canonical tie-break rule.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def compute_k(num_experts: int, num_gpus: int) -> int:
    """Return ceil(num_experts / num_gpus): the per-GPU fetch ceiling K."""
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if num_experts <= 0:
        return 0
    return math.ceil(num_experts / num_gpus)


def evaluate_assignment(D: np.ndarray, assign: np.ndarray) -> dict:
    """Score an expert->GPU assignment (benefit = local demand served)."""
    D = np.asarray(D)
    assign = np.asarray(assign)
    M, G = D.shape
    benefit = int(D[np.arange(M), assign].sum()) if M else 0
    total_tokens = int(D.sum())
    fetch_count = np.bincount(assign, minlength=G).astype(int) if M else np.zeros(G, int)
    return {
        "benefit": benefit,
        "total_tokens": total_tokens,
        "remote_cost": total_tokens - benefit,
        "fetch_count": fetch_count,
        "max_fetch_count": int(fetch_count.max()) if M else 0,
    }


def validate_assignment(D: np.ndarray, assign: np.ndarray, K: int) -> None:
    """Raise AssertionError if the assignment is structurally invalid."""
    D = np.asarray(D)
    assign = np.asarray(assign)
    M, G = D.shape
    assert assign.shape == (M,), f"assign shape {assign.shape} != ({M},)"
    if M == 0:
        return
    assert np.issubdtype(assign.dtype, np.integer), "assign must be integer GPU ids"
    assert assign.min() >= 0 and assign.max() < G, "assigned GPU id out of range"
    fetch_count = np.bincount(assign, minlength=G)
    assert fetch_count.max() <= K, (
        f"fetch_count ceiling violated: max {fetch_count.max()} > K={K}")


# --------------------------------------------------------------------------- #
# Prefill: expanded-slot Hungarian (exact optimal, supports per-GPU capacity)
# --------------------------------------------------------------------------- #
def solve_hungarian(D: np.ndarray, K: Optional[int] = None,
                    caps: Optional[Sequence[int]] = None) -> np.ndarray:
    """Exact optimal expert->GPU assignment via expanded-slot Hungarian.

    Each GPU g is expanded into ``caps[g]`` identical capacity slots, giving a
    ``[M, sum(caps)]`` cost matrix with ``cost[e][slot] = max(D[e]) -
    D[e][gpu(slot)]``.  Subtracting the per-row constant ``max(D[e])`` does not
    change the optimal assignment, so minimising cost == maximising local
    benefit ``sum_e D[e][assign[e]]`` (== minimising NVLink remote movement).

    Args:
        D    : [M, G] integer demand matrix (D[e][g] = tokens on GPU g routing
               to miss expert e).
        K    : uniform per-GPU capacity (used when ``caps`` is None).  Default
               ``compute_k(M, G)``.
        caps : optional per-GPU capacity vector (length G).  Overrides K.

    Returns: ``assign`` int64 array of length M (assign[e] = GPU id).
    """
    D = np.asarray(D)
    M, G = D.shape
    if M == 0:
        return np.zeros(0, dtype=np.int64)
    if caps is None:
        if K is None:
            K = compute_k(M, G)
        caps = np.full(G, int(K), dtype=np.int64)
    else:
        caps = np.asarray(caps, dtype=np.int64)
        assert caps.shape == (G,), f"caps len {caps.shape} != G={G}"
    slot_gpu = np.repeat(np.arange(G), caps)  # length sum(caps); slot -> gpu id
    if slot_gpu.size < M:
        raise ValueError(
            f"total capacity {int(slot_gpu.size)} < num miss experts {M}")
    cost = D.max(axis=1, keepdims=True) - D[:, slot_gpu]
    rows, cols = linear_sum_assignment(cost)
    assign = np.full(M, -1, dtype=np.int64)
    assign[rows] = slot_gpu[cols]
    assert (assign >= 0).all(), "Hungarian failed to assign every expert"
    return assign


# --------------------------------------------------------------------------- #
# Decode: prefix-constrained quota DP (algorithm.md §2.3-2.4)
# --------------------------------------------------------------------------- #
def decode_quota_dp(prefix_cost: List[Sequence[float]], M: int) -> Optional[List[int]]:
    """Choose per-GPU slot quota q_g minimising total prefix eviction cost.

    Args:
        prefix_cost : list (len G) of per-GPU prefix-cost arrays.
                      ``prefix_cost[g][q]`` = cost of opening the first q slots
                      of GPU g (``prefix_cost[g][0] == 0``).  ``q`` ranges
                      ``0..qmax_g`` where ``qmax_g = len(prefix_cost[g]) - 1``.
        M           : total slots to open across all GPUs (== number of misses).

    Returns:
        ``q_g`` list (len G) with ``sum(q_g) == M`` and ``0 <= q_g <= qmax_g``,
        minimising ``sum_g prefix_cost[g][q_g]``.  ``None`` if infeasible
        (``sum_g qmax_g < M``).

    Canonical tie-break (cross-rank determinism): GPUs are processed in index
    order; for each reachable total we keep the FIRST (smallest-cost) value and,
    on equal cost, the smallest q (strict ``<`` while iterating q ascending).
    This is deterministic and identical on every rank.
    """
    G = len(prefix_cost)
    qmax = [len(pc) - 1 for pc in prefix_cost]
    if sum(qmax) < M:
        return None
    INF = float("inf")
    dp = [INF] * (M + 1)
    dp[0] = 0.0
    # choice[i][m] = q chosen at GPU i to land on total m (after processing GPU i)
    choice = [[-1] * (M + 1) for _ in range(G)]
    for i in range(G):
        ndp = [INF] * (M + 1)
        pc_i = prefix_cost[i]
        qm_i = qmax[i]
        for m in range(M + 1):
            base = dp[m]
            if base == INF:
                continue
            top = min(qm_i, M - m)
            for q in range(0, top + 1):
                nm = m + q
                c = base + pc_i[q]
                if c < ndp[nm]:  # strict: ascending q => smallest q kept on tie
                    ndp[nm] = c
                    choice[i][nm] = q
        dp = ndp
    if dp[M] == INF:
        return None
    # backtrack
    q_g = [0] * G
    m = M
    for i in range(G - 1, -1, -1):
        q = choice[i][m]
        q_g[i] = q
        m -= q
    assert m == 0 and sum(q_g) == M
    return q_g


# --------------------------------------------------------------------------- #
# Decode: miss-expert -> selected-slot Hungarian (algorithm.md §3.4)
# --------------------------------------------------------------------------- #
def decode_match(benefit: np.ndarray) -> np.ndarray:
    """Maximise total placement benefit matching M experts to M selected slots.

    Args:
        benefit : [M, M] matrix.  ``benefit[e][j]`` = placement benefit of
                  putting miss expert e into selected slot j (a specific physical
                  slot on a specific GPU).  May be integer (Stage1: D_now) or
                  float (Stage2: current + lambda*future locality).

    Returns: ``match`` int64 array of length M (match[e] = selected-slot column).
    """
    benefit = np.asarray(benefit)
    M = benefit.shape[0]
    if M == 0:
        return np.zeros(0, dtype=np.int64)
    assert benefit.shape == (M, M), f"benefit must be square, got {benefit.shape}"
    # rowmax - benefit: minimising == maximising benefit (per-row constant).
    cost = benefit.max(axis=1, keepdims=True) - benefit
    rows, cols = linear_sum_assignment(cost)
    match = np.full(M, -1, dtype=np.int64)
    match[rows] = cols
    assert (match >= 0).all(), "decode_match failed to match every expert"
    return match
