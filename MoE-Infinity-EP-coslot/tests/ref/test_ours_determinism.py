"""OURS — cross-rank determinism.

Every EP rank runs the identical OURS computation on byte-identical inputs and
must produce byte-identical plans + shadow, else Phase-4 drift FATALs.  We guard
this two ways:

  1. within-process: fresh controllers fed the same trace produce identical
     fetch_ops + shadow every layer (catches accidental RNG / nondeterminism).
  2. cross-process: the same trace run in two subprocesses with different
     PYTHONHASHSEED prints identical digests (catches builtin-hash / set-order
     leakage that would diverge across ranks).

Run:  pytest tests/ref/test_ours_determinism.py -q
"""
from __future__ import annotations

import os
import random
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reference import demand_matrix                      # noqa: E402
import prod_driver as pd                                  # noqa: E402

L = 0
_HERE = os.path.dirname(__file__)


def _gen_trace(rng):
    ep = rng.choice([2, 4])
    ne = rng.choice([12, 16, 24])
    cap = rng.choice([3, 4, 6])
    n_steps = rng.randint(3, 7)
    steps = []
    for _ in range(n_steps):
        k = rng.randint(1, min(ne, 10))
        step = {r: {} for r in range(ep)}
        for e in rng.sample(range(ne), k):
            step[rng.randrange(ep)][e] = rng.randint(1, 20)
        steps.append(step)
    return ep, cap, ne, steps


def _run(prod, steps, ep, ne, is_decode):
    out = []
    for step in steps:
        matrix = demand_matrix(ep, ne, step)
        hits, misses, shadow = pd.prod_plan_tuples(prod, matrix, L, is_decode=is_decode)
        out.append((hits, {r: sorted(misses[r], key=lambda m: m[5]) for r in range(ep)},
                    shadow))
        pd.prod_bump(prod)
    return out


@pytest.mark.parametrize("is_decode", [False, True], ids=["prefill", "decode"])
@pytest.mark.parametrize("seed", range(60))
def test_ours_within_process_determinism(seed, is_decode):
    rng = random.Random(seed * 4099 + 7)
    ep, cap, ne, steps = _gen_trace(rng)
    a = _run(pd.make_prod_ours(ep, cap, ne), steps, ep, ne, is_decode)
    b = _run(pd.make_prod_ours(ep, cap, ne), steps, ep, ne, is_decode)
    c = _run(pd.make_prod_ours(ep, cap, ne), steps, ep, ne, is_decode)
    assert a == b == c, f"OURS nondeterministic seed={seed} decode={is_decode}"


def _subproc_digest(mode, is_decode, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    r = subprocess.run(
        [sys.executable, os.path.join(_HERE, "_det_runner.py"), mode,
         str(int(is_decode))],
        capture_output=True, text=True, env=env, cwd=_HERE)
    assert r.returncode == 0, f"runner failed: {r.stderr}"
    return r.stdout


@pytest.mark.parametrize("is_decode", [False, True], ids=["prefill", "decode"])
def test_ours_cross_process_determinism(is_decode):
    d0 = _subproc_digest("ours", is_decode, 0)
    d1 = _subproc_digest("ours", is_decode, 1)
    d2 = _subproc_digest("ours", is_decode, 12345)
    assert d0 == d1 == d2, "OURS plan depends on PYTHONHASHSEED (hash/set leak)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
