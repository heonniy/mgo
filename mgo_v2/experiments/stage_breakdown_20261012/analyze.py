"""Aggregate the stage-breakdown campaign: per cell mean and std over repetitions.

Per run, decode layers only, in seconds summed over the decode:
  h2d_straggler      = sum_l max_r wait[l, r]
  h2d_imbalance      = sum_l (max_r - mean_r) wait[l, r]
  compute_imbalance  = sum_l (max_r - mean_r) compute[l, r]
  own_imbalance      = sum_l (max_r - mean_r) (wait + compute)[l, r]
  comm_cost          = sum_l [mean_r comm[l, r] - (max_r - mean_r) own[l, r]]  (comm net of waiting for the busiest rank)
  ep_cost            = own_imbalance + comm_cost
plus TPOT (Qwen: the timed, uninstrumented target; DeepSeek: the instrumented teacher target).
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'placement_decomposition_20261012'))
W = 8
QCOMM = ('moe.forward_complete', 'forward_token_a2a_submit', 'moe.forward_a2a',
         'return_token_a2a', 'moe.return_a2a', 'metadata_rank_all_gather')


def qwen_arrays(path):
    ranks = [json.loads((path / f'generation_diagnostic_rank{r}.json').read_text()) for r in range(W)]
    ev = []
    for d in ranks:
        e = defaultdict(lambda: defaultdict(float))
        for row in d['segments']:
            i = row.get('event_index')
            if i is not None and i >= 48:
                e[i][row['phase']] += row['stream_seconds']
        ev.append(e)
    keys = sorted(set.intersection(*[set(e) for e in ev]))
    g = lambda ph: np.array([[ev[r][k].get(ph, 0.0) for r in range(W)] for k in keys])
    tpot = json.loads((path / 'repeat1.json').read_text())['TPOT']
    return g('moe.expert_compute'), g('required_h2d_exposed_wait'), sum(g(p) for p in QCOMM), tpot


def deepseek_arrays(path):
    ranks = [json.loads((path / f'repeat1_rank{r}.json').read_text()) for r in range(W)]
    lay = lambda r, n: np.concatenate([np.asarray(s.get(n + '_layer_ms', []), float)
                                       for s in ranks[r]['diagnostic_steps']]) / 1e3
    g = lambda n: np.stack([lay(r, n) for r in range(W)], 1)
    tpot = json.loads((path / 'repeat1.json').read_text())['TPOT']
    return g('compute'), g('serial_wait'), g('dispatch') + g('return'), tpot


def metrics(comp, wait, comm, tpot):
    own = comp + wait
    oi = own.max(1) - own.mean(1)
    return dict(h2d_straggler=wait.max(1).sum(), h2d_imbalance=(wait.max(1) - wait.mean(1)).sum(),
                mean_wait=wait.mean(1).sum(), mean_compute=comp.mean(1).sum(),
                compute_imbalance=(comp.max(1) - comp.mean(1)).sum(), own_imbalance=oi.sum(),
                comm_cost=(comm.mean(1) - oi).sum(), ep_cost=oi.sum() + (comm.mean(1) - oi).sum(),
                TPOT=tpot)


def main():
    jobs = json.loads((HERE / 'STATUS.json').read_text())['jobs']
    if (HERE / 'STATUS_extra.json').exists():
        jobs += json.loads((HERE / 'STATUS_extra.json').read_text())['jobs']  # unconstrained-quota arms
    cells = defaultdict(list)
    for job in jobs:
        path = Path(job['output'])
        fn = qwen_arrays if job['model'] == 'qwen' else deepseek_arrays
        cells[(job['model'], job['arm'])].append(dict(rep=job['rep'], **metrics(*fn(path))))
    summary = {}
    for (model, arm), runs in sorted(cells.items()):
        summary[f'{model}/{arm}'] = {k: dict(mean=statistics.mean(r[k] for r in runs),
                                             std=statistics.stdev(r[k] for r in runs) if len(runs) > 1 else 0.0,
                                             values=[r[k] for r in runs])
                                     for k in runs[0] if k != 'rep'}
        summary[f'{model}/{arm}']['n'] = len(runs)
    (HERE / 'SUMMARY.json').write_text(json.dumps(summary, indent=1) + '\n')
    cols = ('h2d_straggler', 'h2d_imbalance', 'compute_imbalance', 'comm_cost', 'ep_cost', 'TPOT')
    print('cell'.ljust(16), 'n', *[c.rjust(19) for c in cols])
    for k, v in summary.items():
        print(k.ljust(16), v['n'], *[f"{v[c]['mean']:9.3f} ±{v[c]['std']:7.3f}" for c in cols])


if __name__ == '__main__':
    main()
