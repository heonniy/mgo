"""Count global distinct experts in variable-length prefill/decode events."""
import numpy as np


def distinct_event_counts(rank_routes, sizes):
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    if any(len(row) != offsets[-1] for row in rank_routes):
        raise ValueError('rank routing lengths disagree')
    return np.array([
        len(np.unique(np.concatenate([
            row[offsets[event]:offsets[event + 1]] for row in rank_routes
        ])))
        for event in range(len(sizes))
    ], dtype=np.int64)
