"""SPLIT quota tables for induced-contention levels (calibrated copy-time ratio GPUs0-3 / GPUs4-7).

For each miss count m: pick group totals (QA on GPUs 0-3, QB on 4-7) minimizing
max(r * ceil(QA/4), ceil(QB/4)), ties -> smaller |QB - QA|. Within a group the remainder goes to ranks
in the FAST order (4-7: 7,4,6,5; 0-3: 3,2,1,0), so no single rank always takes every extra copy.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATIOS = {'r129': 1.29, 'r172': 1.72, 'r144': 1.44}  # r144: burst-profiled ratio (paper Table 1c), used without hogs
ORDER_A, ORDER_B = [3, 2, 1, 0], [7, 4, 6, 5]


def spread(total, order):
    base, rem = divmod(total, 4)
    q = {r: base for r in order}
    for r in order[:rem]:
        q[r] += 1
    return q


def table(r):
    rows = []
    for m in range(129):
        best = None
        for qa in range(m + 1):
            qb = m - qa
            key = (max(r * -(-qa // 4), -(-qb // 4)), abs(qb - qa))
            if best is None or key < best[0]:
                best = (key, qa, qb)
        _, qa, qb = best
        q = {**spread(qa, ORDER_A), **spread(qb, ORDER_B)}
        row = [q[i] for i in range(8)]
        assert sum(row) == m
        rows.append(row)
    return rows


for tag, r in RATIOS.items():
    rows = table(r)
    out = dict(status='PASS', physical_gpus=list(range(8)), payload_bytes=9437184, ratio=r,
               rule=__doc__.strip().splitlines()[0], quota_lut=dict(NEAR_SPLIT=rows))
    (HERE / f'SPLIT_{tag}.json').write_text(json.dumps(out) + '\n')
    print(tag, {m: rows[m] for m in (12, 40, 64, 68, 72, 74)})
