"""The decode history accelerator must preserve canonical gate-score ties."""

import numpy as np

from mgo_v2.eviction import GateHistory


def test_compiled_gate_history_matches_reference_bitwise():
    rng = np.random.default_rng(9021)
    reference = GateHistory(4, 128, 128)
    compiled = GateHistory(4, 128, 128)
    # Exercise empty/partial/full windows, exact replacement and a long prefill.
    for step in range(400):
        count = int(rng.choice((0, 1, 16, 64, 128, 256)))
        probs = rng.random((count, 128), dtype=np.float32)
        layer = step % 4
        reference.update(layer, probs)
        compiled.update_compiled(layer, probs)
        np.testing.assert_array_equal(compiled.sums, reference.sums)
        np.testing.assert_array_equal(
            np.asarray(compiled.rows[layer]), np.asarray(reference.rows[layer])
        )
        np.testing.assert_array_equal(
            (compiled.sums[layer] / max(1, len(compiled.rows[layer]))).astype(np.float32),
            (reference.sums[layer] / max(1, len(reference.rows[layer]))).astype(np.float32),
        )
