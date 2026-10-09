"""CPU gate: native controller must match the reference across evictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mgo_v2  # Initialize the package before its legacy Policy adapter.
from env_offload_policy import Policy, seed_rng
from native_pcie_controller import native_quotas, SOURCE
from pcie_quota import quota_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for m in range(257):
        for event in range(8):
            for group in (False, True):
                q = native_quotas(m, group, event)
                np.testing.assert_array_equal(q, quota_vector(m, 4, group, event))
                assert q.sum() == m and q.max() - q.min() <= 1
                if group:
                    assert abs(q[:2].sum() - q[2:].sum()) <= 1
    np.testing.assert_array_equal(native_quotas(6, True, 0), [1, 2, 1, 2])
    rng = np.random.default_rng(20261009)
    inputs = []
    for event in range(384):
        n = (1, 8, 64, 256)[event % 4]
        selected = rng.integers(0, 128, (n, 8), dtype=np.int64)
        if event % 7 == 0:
            selected[:] = np.arange(8)
        weights = rng.random((n, 8), dtype=np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        weights.flat[0] = np.float32(.20)
        origins = rng.integers(0, 4, n, dtype=np.int64)
        gates = rng.random(128, dtype=np.float32)
        if event % 5 == 0:
            gates[:] = 0
        inputs.append((selected, weights, origins, gates))
    similarity = np.zeros((48, 128, 128), np.float32)
    costs = np.array([[0, 1, 5, 6], [2, 0, 7, 5], [5, 6, 0, 2], [7, 5, 1, 0]], np.int64)
    future = np.zeros((128, 4), np.int32)
    states = ('slots', 'owner', 'primary', 'last', 'seen', 'lost', 'birth', 'reuses', 'gates')
    receipts = []
    for policy in (0, 1, 7):
        for mode in ('rank_order', 'group_balanced'):
            for weighted in ((False, True) if policy == 1 else (False,)):
                kwargs = dict(quota_mode=mode, peer_costs=costs if weighted else None)
                ref = Policy([96, 96, 96, 95], similarity, False, policy, 42, **kwargs)
                native = Policy([96, 96, 96, 95], similarity, False, policy, 42, native_pcie=True, **kwargs)
                seed_rng(42)
                evictions = 0
                elapsed = 0
                for event, (selected, weights, origins, gates) in enumerate(inputs):
                    expected = ref.apply(event, selected, weights, origins, gates, future)
                    before = time.perf_counter_ns()
                    actual = native.apply(event, selected, weights, origins, gates, future)
                    elapsed += time.perf_counter_ns() - before
                    for index, (a, b) in enumerate(zip(expected, actual)):
                        try:
                            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
                        except AssertionError as exc:
                            raise AssertionError(f'policy={policy} mode={mode} weighted={weighted} event={event} output={index}: {exc}') from exc
                    for state in states:
                        np.testing.assert_array_equal(getattr(ref, state), getattr(native, state), err_msg=f'{policy}/{mode}/{event}/{state}')
                    evictions += int(actual[-1][23])
                assert evictions > 0
                receipt = dict(policy=policy, quota_mode=mode, weighted=weighted, events=len(inputs),
                               evictions=evictions, adapter_plus_native_mean_us=elapsed / len(inputs) / 1000)
                receipts.append(receipt)
                print(json.dumps(receipt), flush=True)
                native.native_pcie.close()
    result = dict(status='PASS', source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  quota_cases=257 * 8 * 2, policy_cases=receipts, wall_seconds=time.perf_counter()-started)
    (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
