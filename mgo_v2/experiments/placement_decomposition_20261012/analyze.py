"""Decompose stage-2 placement cost into load imbalance and communication (serial + inline, FAST quota).

Per decode layer and rank: own = H2D wait + expert compute. A rank that finishes early waits for
the busiest rank inside the next collective, so

    mean_r(comm) = pure_comm + (max_r(own) - mean_r(own))

and pure comm is estimated as mean_r(comm) - imbalance(own). All values are decode-only sums over
layers in seconds. Qwen spans come from the post-generation diagnostic; DeepSeek spans come from the
instrumented teacher-token target (per-layer CUDA-event spans).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

Q = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
D = Path('/home/hwlee/mgo-results/deepseek_r8_sharegpt_b16_l512_20261012/jobs')
ARMS = ('near', 'rand', 'worst')
NAMES = {'near': 'FAST+Near', 'rand': 'FAST+Random', 'worst': 'FAST+Worst'}
QCOMM = ('moe.forward_complete', 'forward_token_a2a_submit', 'moe.forward_a2a',
         'return_token_a2a', 'moe.return_a2a', 'metadata_rank_all_gather')
W = 8


def qwen(tag):
    path = Q / f'qsd_{tag}_full_v1'
    if not (path / 'generation_diagnostic_rank7.json').exists():
        return None
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
    comp, wait = g('moe.expert_compute'), g('required_h2d_exposed_wait')
    comm = sum(g(p) for p in QCOMM)
    place = ranks[0]['decode_placement_events']
    extra = dict(remote_rows=sum(p['remote_expert_rows'] for p in place),
                 total_rows=sum(p['total_expert_rows'] for p in place),
                 transport_MB=sum(r['decode_transport_bytes']['forward_bytes'] + r['decode_transport_bytes']['return_bytes']
                                  for r in ranks) / 1e6,
                 wall_s=ranks[0]['wall_seconds'])
    return comp, wait, comm, extra


def deepseek(tag):
    path = D / f'dsd_{tag}_full_v1'
    if not (path / 'repeat1_rank7.json').exists():
        return None
    ranks = [json.loads((path / f'repeat1_rank{r}.json').read_text()) for r in range(W)]
    def layers(r, name):
        return np.concatenate([np.asarray(s.get(name + '_layer_ms', []), float) for s in ranks[r]['diagnostic_steps']]) / 1e3
    g = lambda name: np.stack([layers(r, name) for r in range(W)], 1)
    comp, wait = g('compute'), g('serial_wait')
    comm = g('dispatch') + g('return')
    # Decode only: step 0 is the prefill forward.
    extra = dict(transport_MB=sum(s['dispatch_bytes'] + s['return_bytes'] for r in ranks for s in r['diagnostic_steps'] if s['step'] > 0) / 1e6,
                 TPOT=json.loads((path / 'repeat1.json').read_text())['TPOT'])
    return comp, wait, comm, extra


def decompose(comp, wait, comm):
    own = comp + wait
    return dict(mean_compute=comp.mean(1).sum(), compute_imbalance=(comp.max(1) - comp.mean(1)).sum(),
                mean_wait=wait.mean(1).sum(), own_imbalance=(own.max(1) - own.mean(1)).sum(),
                critical_own=own.max(1).sum(), mean_comm=comm.mean(1).sum(),
                pure_comm=(comm.mean(1) - (own.max(1) - own.mean(1))).sum(), layers=own.shape[0])


report = {}
for model, fn in (('Qwen3', qwen), ('DeepSeek', deepseek)):
    for tag in ARMS:
        got = fn(tag)
        if got is None:
            continue
        comp, wait, comm, extra = got
        report[f'{model} {NAMES[tag]}'] = dict(**{k: (round(float(v), 3) if not isinstance(v, int) else v)
                                                for k, v in decompose(comp, wait, comm).items()},
                                               **{k: round(float(v), 4) for k, v in extra.items()})
out = Path(__file__).resolve().parent / 'DECOMPOSITION.json'
out.write_text(json.dumps(report, indent=1) + '\n')
cols = ('mean_compute', 'compute_imbalance', 'mean_wait', 'own_imbalance', 'critical_own', 'mean_comm', 'pure_comm', 'transport_MB')
print('arm'.ljust(24), *[c[:14].rjust(14) for c in cols])
for k, v in report.items():
    print(k.ljust(24), *[f'{v.get(c, float("nan")):14.3f}' for c in cols])
