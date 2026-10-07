"""CPU-only parity of CA local-demand objective and hard rank quotas."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from br_carep_cpu import balanced_assignment
from native_ca_assignment import native_balanced_assignment


def test_native_ca_objective_and_quota_parity():
    rng = np.random.default_rng(20261008)
    for world in (1, 2, 4, 8):
        for n in (0, 1, 2, 7, 30, 60, 90, 128):
            for _ in range(20):
                demand = rng.integers(0, 20, (128, world), dtype=np.int64)
                experts = np.sort(rng.choice(128, n, replace=False)).astype(np.int64)
                legacy = balanced_assignment(demand, experts, world, False)
                native = native_balanced_assignment(demand, experts, world, False)
                assert np.array_equal(np.bincount(native, minlength=world),
                                      np.bincount(legacy, minlength=world))
                assert demand[experts, native].sum() == demand[experts, legacy].sum()
                assert np.array_equal(native, native_balanced_assignment(demand, experts, world, False))
