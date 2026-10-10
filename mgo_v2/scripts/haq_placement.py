"""Hit-aware miss quota (HAQ) for serial-GEMM, non-overlapped decode.

With per-expert serial GEMM and no H2D/compute overlap, one rank's MoE layer
costs about q*(h2d + c) + h*c, where q is its demand-miss count, h its count of
resident (hit) experts used by this layer, h2d the 9 MiB copy time and c the
per-expert serial GEMM time.  The balanced quota equalizes q only; HAQ instead
water-fills q against h, so hit-heavy ranks take fewer misses, while never
exceeding the balanced per-rank maximum ceil(m/world) copies.  Which miss goes
to which rank inside that quota reuses the Near (LA-first, locality-second)
rule.
"""
import numpy as np
from numba import njit
from la_placement import estimate_loads

# Integer microseconds, calibrated on H100 (9 MiB pinned H2D ~180 us; native
# serial expert ~71 us, rows <= 256).
H2D_US = 180
EXPERT_US = 71


@njit(cache=True)
def hit_aware_quota(hits, m, world, h2d_us=H2D_US, expert_us=EXPERT_US):
    cap = (m + world - 1) // world
    quota = np.zeros(world, np.int64)
    cost = np.empty(world, np.int64)
    for r in range(world):
        cost[r] = hits[r] * expert_us
    for _ in range(m):
        best = -1
        for r in range(world):
            if quota[r] >= cap:
                continue
            if best < 0 or cost[r] < cost[best]:
                best = r
        quota[best] += 1
        cost[best] += h2d_us + expert_us
    return quota


@njit(cache=True)
def worst_balanced_quota(hits, m, world):
    # Adversarial control: balanced +/-1 quota, remainder copies on the
    # hit-heaviest ranks (maximizes q*(h2d+c)+h*c under the balanced quota).
    quota = np.full(world, m // world, np.int64)
    order = np.argsort(-hits, kind='mergesort')
    for i in range(m % world):
        quota[order[i]] += 1
    return quota


@njit(cache=True)
def near_assignment_with_quota(demand, misses, owner, layer, quota, near_bps=200):
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
        projected = np.full(world, 2**60, np.int64)
        best_comp = 2**60
        for r in range(world):
            if remaining[r] <= 0:
                continue
            mx = 0
            for rr in range(world):
                v = loads[rr] + (total if rr == r else 0)
                if v > mx:
                    mx = v
            projected[r] = mx
            if mx < best_comp:
                best_comp = mx
        limit = best_comp + max(1, (best_comp * near_bps) // 10000)
        best = -1
        best_local = -1
        best_comp2 = 2**60
        best_dst = 2**60
        for r in range(world):
            if remaining[r] <= 0 or projected[r] > limit:
                continue
            local = demand[e, r]
            dst = loads[r] + total
            if (local > best_local or
                (local == best_local and projected[r] < best_comp2) or
                (local == best_local and projected[r] == best_comp2 and dst < best_dst)):
                best = r; best_local = local; best_comp2 = projected[r]; best_dst = dst
        assert best >= 0
        answer[i] = best
        remaining[best] -= 1
        loads[best] += total
    return answer
