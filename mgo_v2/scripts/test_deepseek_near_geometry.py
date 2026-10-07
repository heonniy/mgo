"""Compiled Near admission must support both Qwen and DeepSeek geometries."""
import os

# A fresh cache is needed when changing a transitive Numba helper: cached
# callers do not automatically track edits to estimate_loads().
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/mgo-main-table-deepseek-numba-test')

import numpy as np
import mgo_v2  # Establish the package import before env_offload_policy.
from env_offload_policy import Policy


def check(layers, experts, topk, capacities):
    policy = Policy(capacities, np.zeros((layers, experts, experts), np.float32), False, 7)
    selected = np.tile(np.arange(topk, dtype=np.int16), (8, 1))
    weights = np.full((8, topk), 1 / topk, np.float32)
    origins = np.repeat(np.arange(4), 2).astype(np.int64)
    gates = np.full(experts, 1 / experts, np.float32)
    result = policy.apply(0, selected, weights, origins, gates, np.zeros((experts, 4), np.int32))
    assert len(result[5]) == topk
    assert result[1].shape == selected.shape
    assert result[4].shape == selected.shape
    assert np.count_nonzero(policy.owner) == topk


if __name__ == '__main__':
    check(48, 128, 8, [459, 459, 459, 458])
    check(26, 64, 6, [123, 123, 123, 122])
    print('PASS Qwen and DeepSeek compiled Near geometries')
