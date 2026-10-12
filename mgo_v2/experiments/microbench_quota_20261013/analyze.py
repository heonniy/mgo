"""Qwen microbench-quota campaign in ms per decode step (stage_breakdown metrics) plus the reused random quota.

Also checks every decode event's realized quota spread (max_r - min_r fetches) from decode_placement_events.
"""
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SB = HERE.parent / 'stage_breakdown_20261012'
sys.path.insert(0, str(SB))
from analyze import metrics, qwen_arrays  # noqa: E402

STEPS = 63
NAMES = {'near': 'NEAR', 'fast': 'FAST', 'mbbal': 'MB_BAL', 'mbopt': 'MB_OPT', 'bw': 'BRAND_W', 'rquota': 'RANDOM_QUOTA'}
COLS = ('h2d_straggler', 'h2d_imbalance', 'own_imbalance', 'comm_cost', 'ep_cost', 'TPOT')


def spread(path):
    ev = json.loads((Path(path) / 'generation_diagnostic_rank0.json').read_text())['decode_placement_events']
    s = [max(e['rank_fetches']) - min(e['rank_fetches']) for e in ev]
    return dict(events=len(s), events_spread_gt1=sum(x > 1 for x in s), max_spread=max(s))


def main():
    jobs = json.loads((HERE / 'STATUS.json').read_text())['jobs']
    jobs += [j for j in json.loads((SB / 'STATUS_extra.json').read_text())['jobs'] if j['model'] == 'qwen' and j['arm'] == 'rquota']
    cells = defaultdict(list); rows = []
    for j in jobs:
        m = {k: float(v) * 1e3 / (1 if k == 'TPOT' else STEPS) for k, v in metrics(*qwen_arrays(Path(j['output']))).items()}
        m.update(spread(j['output']))
        cells[j['arm']].append(m)
        rows.append(dict(arm=NAMES[j['arm']], rep=j['rep'], **{k: round(v, 4) for k, v in m.items()}))
    with open(HERE / 'RUNS_per_step.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for arm in NAMES:
        if arm in cells:
            r = cells[arm]
            summ[NAMES[arm]] = {k: dict(mean=statistics.mean(x[k] for x in r), std=statistics.stdev(x[k] for x in r) if len(r) > 1 else 0.0)
                                for k in COLS} | dict(n=len(r), events_spread_gt1=sum(x['events_spread_gt1'] for x in r),
                                                       events=sum(x['events'] for x in r), max_spread=max(x['max_spread'] for x in r))
    (HERE / 'SUMMARY.json').write_text(json.dumps(summ, indent=1) + '\n')
    base = summ.get('NEAR')
    print('arm'.ljust(13), 'n', *[c.rjust(18) for c in COLS], ' TPOT/NEAR  spread>1 events (max)')
    for k, v in summ.items():
        d = f"{100 * (v['TPOT']['mean'] / base['TPOT']['mean'] - 1):+6.2f}%" if base else ''
        print(k.ljust(13), v['n'], *[f"{v[c]['mean']:9.2f} ±{v[c]['std']:6.2f}" for c in COLS], d,
              f"{v['events_spread_gt1']}/{v['events']} ({v['max_spread']})")


if __name__ == '__main__':
    main()
