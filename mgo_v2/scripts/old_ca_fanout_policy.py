"""Original a0be82e communication-only Hungarian objective, for CPU PLAN."""
import numpy as np
from numba import njit, objmode
from scipy.optimize import linear_sum_assignment


@njit(cache=True)
def incremental_costs(effective, lengths, origins, primary, layer, experts,
                      incoming, world):
    # Snapshot destinations before admissions. Incoming assignments never
    # update this base: the historical objective is linearized, not greedy.
    costs = np.zeros((len(incoming), world), np.int64)
    positions = np.full(experts, -1, np.int64)
    for i in range(len(incoming)):
        positions[incoming[i]] = i
    for t in range(len(origins)):
        base = 0
        for j in range(lengths[t]):
            rank = primary[layer * experts + effective[t, j]]
            if rank >= 0:
                base |= 1 << rank
        for j in range(lengths[t]):
            i = positions[effective[t, j]]
            if i >= 0:
                for rank in range(world):
                    if rank != origins[t] and not (base & (1 << rank)):
                        costs[i, rank] += 1
    return costs


@njit(cache=True)
def fanout_assignment(effective, lengths, origins, primary, layer, experts,
                      incoming, world):
    n = len(incoming)
    slots = np.empty(n, np.int64)
    at = 0
    for rank in range(world):
        for _ in range(n // world + (rank < n % world)):
            slots[at] = rank
            at += 1
    if n == 0:
        return slots
    costs = incremental_costs(effective, lengths, origins, primary, layer,
                              experts, incoming, world)
    # Use the historical SciPy solver and slot order, including tie behavior.
    with objmode(answer='int64[:]'):
        rows, cols = linear_sum_assignment(costs[:, slots])
        answer = np.empty(n, np.int64)
        answer[rows] = slots[cols]
    return answer
