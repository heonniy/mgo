"""Pilot-only balanced miss admission: critical miss rows, then packets."""

import numpy as np
from numba import njit

from .fanout_admission import (
    _balanced_quotas, _prepare_packet_state, _candidate_packet_cost,
    _commit_packet_choice,
)


@njit(cache=True)
def assign_miss_quota(demand, effective, lengths, origins, primary,
                      layer, experts, misses, world, worst):
    """Return one destination per miss with identical hard quotas.

    The candidate first minimizes the projected maximum *miss* expert rows
    among ranks with the largest miss quota, then among all ranks. Only ties
    use exact token/destination packet cost and local demand. BW deliberately
    puts the hottest misses on the first maximum-quota rank.
    """
    m = len(misses)
    quota = _balanced_quotas(m, world)
    remaining = quota.copy()
    answer = np.empty(m, np.int64)
    if m == 0:
        return answer
    totals = np.zeros(m, np.int64)
    for i in range(m):
        for r in range(world):
            totals[i] += demand[misses[i], r]
    order = np.argsort(-totals)
    presence, masks, incident = _prepare_packet_state(
        effective, lengths, origins, primary, layer, experts, misses, world)
    miss_rows = np.zeros(world, np.int64)
    max_quota = np.max(quota)
    target = 0
    while quota[target] != max_quota:
        target += 1
    for oi in range(m):
        i = int(order[oi])
        e = int(misses[i])
        load = int(totals[i])
        best = -1
        best_qmax = 2**60
        best_allmax = 2**60
        best_packets = 2**60
        best_delta = 2**60
        best_local = -1
        for r in range(world):
            if remaining[r] <= 0:
                continue
            if worst:
                # Target receives the hottest experts until its quota fills.
                if remaining[target] > 0 and r != target:
                    continue
                if remaining[target] == 0 and r == target:
                    continue
                best = r
                break
            qmax = 0
            allmax = 0
            for rr in range(world):
                projected = miss_rows[rr] + (load if rr == r else 0)
                if projected > allmax:
                    allmax = projected
                if quota[rr] == max_quota and projected > qmax:
                    qmax = projected
            packets, delta = _candidate_packet_cost(
                i, r, presence, masks, origins, incident, world)
            local = int(demand[e, r])
            if (qmax < best_qmax or
                (qmax == best_qmax and allmax < best_allmax) or
                (qmax == best_qmax and allmax == best_allmax and packets < best_packets) or
                (qmax == best_qmax and allmax == best_allmax and packets == best_packets and delta < best_delta) or
                (qmax == best_qmax and allmax == best_allmax and packets == best_packets and delta == best_delta and local > best_local)):
                best = r
                best_qmax = qmax
                best_allmax = allmax
                best_packets = packets
                best_delta = delta
                best_local = local
        assert best >= 0
        answer[i] = best
        remaining[best] -= 1
        miss_rows[best] += load
        _commit_packet_choice(i, best, presence, masks, origins, incident)
    assert remaining.sum() == 0
    return answer
