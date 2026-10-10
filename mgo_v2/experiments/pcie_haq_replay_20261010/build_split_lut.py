"""Build the NEAR_SPLIT quota table (129x8) from measured group-uniform H2D grid.

For m >= 32 misses: choose group totals (QA on GPUs 0-3, QB on 4-7), each spread
evenly inside its group (max-min <= 1), minimizing predicted max-rank H2D time from
linear fits to grid_bench.json (all eight GPUs active). Ties prefer the split closest
to balanced. For m < 32 copy the existing NEAR_FAST row (balanced, extras to 4-7).
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
grid = json.loads((HERE / 'grid_bench.json').read_text())['cases']
X, YA, YB = [], [], []
for v in grid.values():
    q, ms = v['quota'], v['rank_median_ms']
    X.append((q[0], q[4])); YA.append(np.mean(ms[:4])); YB.append(np.mean(ms[4:]))
X = np.array(X, float)
ca = np.linalg.lstsq(np.c_[np.ones(len(X)), X[:, 0], X[:, 1]], YA, rcond=None)[0]
cb = np.linalg.lstsq(np.c_[np.ones(len(X)), X[:, 1], X[:, 0]], YB, rcond=None)[0]


def pred(qa, qb):
    ta = ca[0] + ca[1] * qa + ca[2] * qb if qa else 0.0
    tb = cb[0] + cb[1] * qb + cb[2] * qa if qb else 0.0
    return max(ta, tb)


def spread(total, ranks):
    base, rem = divmod(total, len(ranks))
    return [base + (i < rem) for i in range(len(ranks))]


fast = json.loads((HERE.parent / 'pcie_quota_r8_b16_20261010/LOOKUP.json').read_text())['quota_lut']['NEAR_FAST']
lut = []
for m in range(129):
    if m < 32:
        lut.append(list(fast[m])); continue
    best = None
    for QA in range(0, m + 1):
        QB = m - QA
        a, b = spread(QA, range(4)), spread(QB, range(4))
        key = (round(pred(max(a), max(b)), 4), abs(QB - QA))
        if best is None or key < best[0]:
            best = (key, a + b)
    lut.append(best[1])
for m, row in enumerate(lut):
    assert sum(row) == m and len(row) == 8
out = dict(status='PASS', physical_gpus=list(range(8)), payload_bytes=9437184,
           model=dict(A=ca.tolist(), B=cb.tolist(), source='grid_bench.json'),
           quota_lut=dict(NEAR_SPLIT=lut, NEAR_FAST=fast))
(HERE / 'SPLIT_LOOKUP.json').write_text(json.dumps(out, indent=1) + '\n')
for m in (32, 40, 48, 56, 60, 64, 68, 69, 72, 74, 80, 88, 96, 112, 128):
    print(m, lut[m])
