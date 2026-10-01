"""Unit tests for the pure OURS solvers vs brute-force optima.

  solve_hungarian   — exact max-benefit assignment under per-GPU capacity.
  decode_quota_dp   — exact min prefix-eviction-cost quota selection.
  decode_match      — exact max-benefit expert<->slot matching.

Run:  pytest tests/ref/test_ours_solver_unit.py -q
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from moe_infinity_ep.controller import assignment as A  # noqa: E402


def _brute_hungarian(D, K):
    M, G = D.shape
    best = -1
    for assign in itertools.product(range(G), repeat=M):
        if np.bincount(assign, minlength=G).max() > K:
            continue
        best = max(best, sum(int(D[e, assign[e]]) for e in range(M)))
    return best


def _brute_quota(prefix, M):
    G = len(prefix)
    qmax = [len(p) - 1 for p in prefix]
    best = float("inf")
    for q in itertools.product(*[range(qm + 1) for qm in qmax]):
        if sum(q) != M:
            continue
        best = min(best, sum(prefix[g][q[g]] for g in range(G)))
    return best


def _brute_match(B):
    M = B.shape[0]
    best = -1
    for perm in itertools.permutations(range(M)):
        best = max(best, sum(int(B[e, perm[e]]) for e in range(M)))
    return best


def test_compute_k():
    assert A.compute_k(4, 2) == 2
    assert A.compute_k(3, 2) == 2
    assert A.compute_k(1, 4) == 1
    assert A.compute_k(0, 4) == 0


def test_hungarian_vs_brute():
    rng = np.random.default_rng(1)
    for _ in range(300):
        M = int(rng.integers(1, 7))
        G = int(rng.integers(1, 4))
        D = rng.integers(0, 9, size=(M, G))
        K = A.compute_k(M, G)
        a = A.solve_hungarian(D, K)
        A.validate_assignment(D, a, K)
        assert int(A.evaluate_assignment(D, a)["benefit"]) == _brute_hungarian(D, K)
        # determinism
        assert np.array_equal(a, A.solve_hungarian(D, K))


def test_hungarian_nonuniform_caps():
    D = np.array([[5, 1], [4, 1], [3, 1]])
    a = A.solve_hungarian(D, caps=[2, 1])
    assert np.bincount(a, minlength=2)[0] <= 2
    assert np.bincount(a, minlength=2)[1] <= 1


def test_hungarian_empty():
    a = A.solve_hungarian(np.zeros((0, 3)), K=1)
    assert a.shape == (0,)


def test_quota_dp_vs_brute():
    rng = np.random.default_rng(2)
    for _ in range(400):
        G = int(rng.integers(1, 4))
        qmax = [int(rng.integers(0, 4)) for _ in range(G)]
        if sum(qmax) == 0:
            continue
        M = int(rng.integers(0, sum(qmax) + 1))
        prefix = []
        for g in range(G):
            steps = rng.integers(0, 5, size=qmax[g])
            pc = [0.0]
            for s in steps:
                pc.append(pc[-1] + float(s))
            prefix.append(pc)
        q = A.decode_quota_dp(prefix, M)
        bc = _brute_quota(prefix, M)
        assert q is not None and sum(q) == M
        got = sum(prefix[g][q[g]] for g in range(G))
        assert abs(got - bc) < 1e-9
        assert all(0 <= q[g] <= qmax[g] for g in range(G))
        assert A.decode_quota_dp(prefix, M) == q  # determinism


def test_quota_dp_infeasible():
    assert A.decode_quota_dp([[0.0, 1.0], [0.0, 1.0]], 5) is None


def test_match_vs_brute():
    rng = np.random.default_rng(3)
    for _ in range(300):
        M = int(rng.integers(1, 7))
        B = rng.integers(0, 9, size=(M, M))
        m = A.decode_match(B)
        assert sorted(m.tolist()) == list(range(M))  # bijection
        assert sum(int(B[e, m[e]]) for e in range(M)) == _brute_match(B)
        assert np.array_equal(m, A.decode_match(B))  # determinism


def test_match_float():
    B = np.array([[0.5, 0.25], [0.25, 0.75]])
    m = A.decode_match(B)
    assert sorted(m.tolist()) == [0, 1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
