"""Summarize an interleaved R8 arm comparison from <experiment>/STATUS.json (first arm = baseline).

Usage: python summarize_arms.py <experiment_dir> [baseline_label]
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TEACHER = json.loads((Path(__file__).resolve().parent.parent /
                      'pcie_quota_r8_b16_20261010/FROZEN_TOKENS.json').read_text())
METRICS = ('TTFT', 'TPOT', 'E2E')


def main():
    exp = Path(sys.argv[1]).resolve()
    status = json.loads((exp / 'STATUS.json').read_text())
    rows = defaultdict(list)
    order = []
    for job in status['jobs']:
        path = Path(job['output'])
        st = json.loads((path / 'status.json').read_text())
        assert st['status'] == 'PASS'
        label = job['label']
        if label not in order:
            order.append(label)
        for rep in range(1, st['repeats'] + 1):
            sample = json.loads((path / f'repeat{rep}.json').read_text())
            ranks = [json.loads((path / f'repeat{rep}_rank{r}.json').read_text()) for r in range(8)]
            assert all(x['phase'] == 'target' and x['forced_continuation'] and x['validation']['status'] == 'PASS'
                       and x['validation']['controller']['quota_violations'] == 0 for x in ranks)
            wait = [x['grouped_executor_counts']['serial_wait_wall_ns'] / 1e9 for x in ranks]
            rows[label].append(dict(attempt=job['attempt'], commit=st['source_commit'],
                                    copies=[x['validation']['scheduler']['copies'] for x in ranks],
                                    waitA=max(wait[:4]), waitB=max(wait[4:]),
                                    **{m: sample[m] for m in METRICS}))
    base = sys.argv[2] if len(sys.argv) > 2 else order[0]
    bmed = {m: statistics.median(r[m] for r in rows[base]) for m in METRICS}
    lines = [f'Baseline: {base}', '',
             '| Arm | n | TPOT median [range], s | vs base | per-round vs base | E2E vs base | max wait 0-3 / 4-7, s | fetches |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    out = {}
    for label in order:
        rs = rows[label]
        med = {m: statistics.median(r[m] for r in rs) for m in METRICS}
        paired = []
        for a in sorted({r['attempt'] for r in rs}):
            mine = [r['TPOT'] for r in rs if r['attempt'] == a]
            ref = [r['TPOT'] for r in rows[base] if r['attempt'] == a]
            if mine and ref:
                paired.append(100 * (statistics.mean(mine) / statistics.mean(ref) - 1))
        tp = [r['TPOT'] for r in rs]
        out[label] = dict(n=len(rs), median=med, vs_base_pct={m: 100 * (med[m] / bmed[m] - 1) for m in METRICS},
                          paired_TPOT_pct=paired, samples=rs)
        lines.append(f"| {label} | {len(rs)} | {med['TPOT']:.4f} [{min(tp):.4f}, {max(tp):.4f}] | "
                     f"{out[label]['vs_base_pct']['TPOT']:+.2f}% | {', '.join(f'{x:+.2f}%' for x in paired)} | "
                     f"{out[label]['vs_base_pct']['E2E']:+.2f}% | "
                     f"{statistics.median(r['waitA'] for r in rs):.2f} / {statistics.median(r['waitB'] for r in rs):.2f} | "
                     f"{int(statistics.median(sum(r['copies']) for r in rs))} |")
    (exp / 'RESULTS.json').write_text(json.dumps(out, indent=1) + '\n')
    (exp / 'RESULTS_TABLE.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
