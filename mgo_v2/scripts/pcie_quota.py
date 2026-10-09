"""Deterministic quotas in rank order GPU0, GPU1, GPU4, GPU5."""
import numpy as np
from numba import njit


@njit(cache=True)
def quota_vector(m, world=4, group_balanced=False, event=0):
    assert m >= 0 and world > 0
    result = np.empty(world, np.int64)
    if not group_balanced:
        for rank in range(world):
            result[rank] = m // world + (rank < m % world)
        return result
    assert world == 4
    for group in range(2):
        total = m // 2 + (m % 2 and group == event % 2)
        priority = (event // 2 + 1) % 2
        for within in range(2):
            result[2 * group + within] = total // 2 + (total % 2 and within == priority)
    assert result.sum() == m
    return result
