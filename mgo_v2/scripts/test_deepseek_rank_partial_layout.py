"""Check compiled dispatch/return indices for six- and eight-route models."""
import os

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/mgo-main-table-deepseek-layout-test')

import numpy as np
from env_offload_layout import plan_rank_partial_layout
from env_offload_rank_layout import layer_physical_slots


def check(topk, experts):
    counts = [2, 2, 2, 2]
    origins = np.repeat(np.arange(4, dtype=np.int64), 2)
    effective = np.tile(np.arange(topk, dtype=np.int16), (8, 1))
    lengths = np.full(8, topk, np.int8)
    destinations = np.empty_like(effective)
    for token in range(8):
        for route in range(topk):
            destinations[token, route] = (token + route) % 4
    layouts = [plan_rank_partial_layout(effective, lengths, destinations, origins, counts, rank, experts)
               for rank in range(4)]
    for rank, layout in enumerate(layouts):
        assert layout['send_eids'].shape[1] == topk
        assert sum(layout['send_counts']) == len(layout['send_idx'])
        assert all(0 <= expert < experts for expert, _, _ in layout['groups'])
        for peer in range(4):
            assert layout['send_counts'][peer] == layouts[peer]['recv_counts'][rank]
    slots = np.array([experts + 2, 2 * experts + 4, -1], np.int32)
    physical = np.array([6, 7, 8], np.int32)
    lookup = layer_physical_slots(slots, physical, 1, experts)
    assert len(lookup) == experts and lookup[2] == 6 and lookup[4] == -1


if __name__ == '__main__':
    check(8, 128)
    check(6, 64)
    print('PASS Qwen and DeepSeek compiled rank-partial layouts')
