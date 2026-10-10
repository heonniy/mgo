"""Stage-2 placement controls inside a fixed per-rank miss quota.

worst_assignment_with_quota mirrors load_locality_near_assignment:
Near minimizes the projected critical-rank expert-row load and, within near_bps
of that optimum, maximizes rank-local demand. The worst control maximizes the
projected critical load (compute imbalance) and, within near_bps of that
maximum, minimizes rank-local demand (remote traffic). Same heavy-first order,
same load estimate, same quota.

random_assignment_with_quota draws a uniformly random rank for each miss
within the quota. Policy.apply seeds the NumPy stream per event.
"""
import numpy as np
from numba import njit
from la_placement import estimate_loads


@njit(cache=True)
def worst_assignment_with_quota(demand, misses, owner, layer, quota, near_bps=200):
    world = demand.shape[1]
    m = len(misses)
    assert quota.sum() == m
    remaining = quota.copy()
    loads = estimate_loads(demand, owner, layer)
    totals = np.empty(m, np.int64)
    for i in range(m):
        totals[i] = demand[misses[i]].sum()
    order = np.argsort(-totals)
    answer = np.empty(m, np.int64)
    for oi in range(m):
        i = order[oi]
        e = misses[i]
        total = totals[i]
        projected = np.full(world, -1, np.int64)
        worst_comp = -1
        for r in range(world):
            if remaining[r] <= 0:
                continue
            mx = 0
            for rr in range(world):
                v = loads[rr] + (total if rr == r else 0)
                if v > mx:
                    mx = v
            projected[r] = mx
            if mx > worst_comp:
                worst_comp = mx
        slack = max(1, (worst_comp * near_bps) // 10000)
        limit = worst_comp - slack
        best = -1
        best_local = 2**60
        best_comp2 = -1
        best_dst = -1
        for r in range(world):
            if remaining[r] <= 0 or projected[r] < limit:
                continue
            local = demand[e, r]
            dst = loads[r] + total
            if (local < best_local or
                    (local == best_local and projected[r] > best_comp2) or
                    (local == best_local and projected[r] == best_comp2 and dst > best_dst) or
                    (local == best_local and projected[r] == best_comp2 and dst == best_dst and (best < 0 or r < best))):
                best = r; best_local = local; best_comp2 = projected[r]; best_dst = dst
        assert best >= 0
        answer[i] = best
        remaining[best] -= 1
        loads[best] += total
    assert remaining.sum() == 0
    return answer


@njit(cache=True)
def random_assignment_with_quota(misses, quota):
    m = len(misses)
    assert quota.sum() == m
    slots = np.empty(m, np.int64)
    k = 0
    for r in range(len(quota)):
        for _ in range(quota[r]):
            slots[k] = r
            k += 1
    return slots[np.random.permutation(m)]
