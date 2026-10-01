"""Helper run as a subprocess by the determinism tests.

Builds a fixed (hash-independent) trace and runs it through a production
controller in the requested mode, printing a JSON digest of the per-layer plan
(fetch tuples per rank + shadow).  Two invocations with different
PYTHONHASHSEED must print identical digests — that proves the plan does not
depend on Python's per-process hash salt (builtin ``hash`` / set iteration
order), which would otherwise diverge across EP ranks and trip Phase-4 drift.

Usage:  python _det_runner.py <mode> <is_decode 0|1>
        mode in {ours, random, naive, balanced}
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from reference import demand_matrix  # noqa: E402
import prod_driver as pd             # noqa: E402

L = 0


def build_trace():
    rng = random.Random(12345)  # PRNG: independent of PYTHONHASHSEED
    ep, cap, ne, nsteps = 4, 6, 24, 6
    steps = []
    for _ in range(nsteps):
        step = {r: {} for r in range(ep)}
        k = rng.randint(4, 12)
        for e in rng.sample(range(ne), k):
            step[rng.randrange(ep)][e] = rng.randint(1, 20)
        steps.append(step)
    return ep, cap, ne, steps


def main():
    mode = sys.argv[1]
    is_decode = bool(int(sys.argv[2]))
    ep, cap, ne, steps = build_trace()
    if mode == "ours":
        prod = pd.make_prod_ours(ep, cap, ne)
    else:
        evict = "random" if mode == "random" else "lfu"
        prod = pd.make_prod(ep, cap, ne, mode, evict)

    digest = []
    for step in steps:
        matrix = demand_matrix(ep, ne, step)
        hits, misses, shadow = pd.prod_plan_tuples(prod, matrix, L, is_decode=is_decode)
        # sort within rank by order for a canonical, comparable representation
        miss_repr = {r: sorted(misses[r], key=lambda m: m[5]) for r in range(ep)}
        shadow_repr = {r: shadow[r] for r in range(ep)}
        digest.append({"hits": hits, "misses": miss_repr, "shadow": shadow_repr})
        pd.prod_bump(prod)
    # JSON with sorted keys; tuples become lists (stable across processes)
    print(json.dumps(digest, sort_keys=True, default=list))


if __name__ == "__main__":
    main()
