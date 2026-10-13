"""Aggregate the induced-contention campaign: per (model, level, arm) mean/std over reps, in ms per decode step.

Metrics are the stage_breakdown ones (decode layers summed, divided by 63 decode steps), plus TPOT.
Deltas are relative to NEAR at the same model and level.
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'stage_breakdown_20261012'))
from analyze import deepseek_arrays, metrics, qwen_arrays  # noqa: E402

STEPS = 63
COLS = ('h2d_straggler', 'h2d_imbalance', 'own_imbalance', 'comm_cost', 'ep_cost', 'TPOT')


def main():
    jobs = [j for f in ('STATUS.json', 'STATUS2.json', 'STATUS3.json') if (HERE / f).exists() for j in json.loads((HERE / f).read_text())['jobs']]
    cells = defaultdict(list); rows = []
    for j in jobs:
        fn = qwen_arrays if j['model'] == 'qwen' else deepseek_arrays
        m = metrics(*fn(Path(j['output'])))
        m = {k: float(v) * 1e3 / (STEPS if k != 'TPOT' else 1) for k, v in m.items()}  # ms/step; TPOT s -> ms
        cells[(j['model'], j['level'], j['arm'])].append(m)
        rows.append(dict(model=j['model'], level=j['level'], arm=j['arm'], rep=j['rep'], **{k: round(v, 4) for k, v in m.items()}))
    with open(HERE / 'RUNS_per_step.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for key, runs in sorted(cells.items()):
        summ['/'.join(key)] = {k: dict(mean=statistics.mean(r[k] for r in runs),
                                       std=statistics.stdev(r[k] for r in runs) if len(runs) > 1 else 0.0)
                               for k in runs[0]} | {'n': len(runs)}
    (HERE / 'SUMMARY.json').write_text(json.dumps(summ, indent=1) + '\n')
    print('cell'.ljust(22), 'n', *[c.rjust(20) for c in COLS], 'TPOT vs NEAR')
    for k, v in summ.items():
        model, level, arm = k.split('/')
        base = summ.get(f'{model}/{level}/near')
        d = f"{100 * (v['TPOT']['mean'] / base['TPOT']['mean'] - 1):+6.2f}%" if base else ''
        print(k.ljust(22), v['n'], *[f"{v[c]['mean']:10.3f} ±{v[c]['std']:7.3f}" for c in COLS], d)


if __name__ == '__main__':
    main()
