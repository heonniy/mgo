"""Where does stage-2 placement change time? Per decode layer and rank: grouped GEMM, a2a, H2D wait.

Spans are current-stream completion spans, so a rank waiting for peers shows the wait
inside a2a/all-gather phases. The per-layer critical rank is the rank with the largest
own work (exposed H2D wait + grouped GEMM).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
ARMS = ('d2_fastnear', 'd2_fastrand', 'd2_fastworst')
W = 8
PH = ('expert_grouped_gemm_kernels', 'required_h2d_exposed_wait', 'forward_token_a2a_submit',
      'moe.forward_complete', 'return_token_a2a', 'metadata_rank_all_gather', 'attention_dense_residual')


def per_event(rank):
    out = defaultdict(lambda: defaultdict(float))
    for row in rank['segments']:
        e = row.get('event_index')
        if e is not None and e >= 48:
            out[e][row['phase']] += row['stream_seconds']
    return out


report = {}
for arm in ARMS:
    path = RAW / f'{arm}_full_v1'
    if not (path / 'generation_diagnostic_rank0.json').exists():
        continue
    ranks = [json.loads((path / f'generation_diagnostic_rank{r}.json').read_text()) for r in range(W)]
    ev = [per_event(r) for r in ranks]
    keys = sorted(set.intersection(*[set(e) for e in ev]))
    get = lambda p: np.array([[ev[r][k].get(p, 0.0) for r in range(W)] for k in keys])
    gemm, wait = get('expert_grouped_gemm_kernels'), get('required_h2d_exposed_wait')
    own = gemm + wait
    place = ranks[0]['decode_placement_events']
    rep = dict(
        wall_s=ranks[0]['wall_seconds'],
        decode_s_per_phase_mean_over_ranks={p: round(float(get(p).sum(0).mean()), 3) for p in PH},
        gemm_sum_over_layers_of_max_s=round(float(gemm.max(1).sum()), 3),
        gemm_sum_over_layers_of_mean_s=round(float(gemm.mean(1).sum()), 3),
        own_sum_over_layers_of_max_s=round(float(own.max(1).sum()), 3),
        remote_expert_rows=int(sum(p['remote_expert_rows'] for p in place)),
        total_expert_rows=int(sum(p['total_expert_rows'] for p in place)),
        remote_rank_packets=int(sum(p['remote_rank_packets'] for p in place)),
        transport_bytes=ranks[0]['decode_transport_bytes'],
        transport_bytes_per_rank=[r['decode_transport_bytes'] for r in ranks])
    report[arm] = rep
out = Path(__file__).resolve().parent / 'DIAG_ANALYSIS.json'
out.write_text(json.dumps(report, indent=1) + '\n')
for arm, rep in report.items():
    print('=====', arm)
    for k, v in rep.items():
        if k != 'transport_bytes_per_rank':
            print(f'  {k}: {v}')
