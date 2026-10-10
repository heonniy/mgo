"""Root-cause analysis of NEAR / FAST / NEAR_SPLIT from post-generation diagnostics.

Inputs: jobs/diag_{near,fast,split}_full_v1/generation_diagnostic_rank{r}.json.
Diagnostic spans are current-stream completion spans (they include host gaps and
peer waits); H2D copy service runs on a separate stream and is reported apart.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
POLICIES = ('near', 'fast', 'split')
W = 8


def load(label):
    return [json.loads((RAW / f'diag_{label}_full_v1' / f'generation_diagnostic_rank{r}.json').read_text())
            for r in range(W)]


def decode_events(rank):
    """Per decode event (event_index >= 48): seconds per phase on this rank."""
    out = defaultdict(lambda: defaultdict(float))
    for row in rank['segments']:
        e = row.get('event_index')
        if e is None or e < 48:
            continue
        out[e][row['phase']] += row['stream_seconds']
    return out


def main():
    report = {}
    for label in POLICIES:
        if not (RAW / f'diag_{label}_full_v1' / 'generation_diagnostic_rank0.json').exists():
            continue
        ranks = load(label)
        copies = [[c for c in r['h2d_copies'] if c['event_index'] is not None and c['event_index'] >= 48] for r in ranks]
        svc = [np.array([c['service_seconds'] for c in cs]) * 1e3 for cs in copies]
        dep = [np.array([c['stream_dependency_seconds'] for c in cs]) * 1e3 for cs in copies]
        sub = [np.array([c['enqueue_to_submit_seconds'] for c in cs]) * 1e3 for cs in copies]
        # Per-event per-rank H2D service sum (one copy stream: serialized service).
        ev_svc = [defaultdict(float) for _ in range(W)]
        for r, cs in enumerate(copies):
            for c in cs:
                ev_svc[r][c['event_index']] += c['service_seconds']
        events = [decode_events(r) for r in ranks]
        keys = sorted(set.intersection(*[set(e) for e in events]))
        phases = sorted({p for e in events for k in e for p in e[k]})
        tot = {p: np.array([sum(events[r][k].get(p, 0.0) for k in keys) for r in range(W)]) for p in phases}
        wait = np.array([[events[r][k].get('required_h2d_exposed_wait', 0.0) for r in range(W)] for k in keys])
        gemm = np.array([[events[r][k].get('expert_grouped_gemm_kernels', 0.0) for r in range(W)] for k in keys])
        own = wait + gemm
        hsv = np.array([[ev_svc[r].get(k, 0.0) for r in range(W)] for k in keys])
        crit = own.argmax(1)
        rep = dict(
            copies_per_rank=[len(c) for c in copies],
            dma_ms_median_per_rank=[round(float(np.median(s)), 4) for s in svc],
            dma_ms_mean_A_B=[round(float(np.concatenate(svc[:4]).mean()), 4), round(float(np.concatenate(svc[4:]).mean()), 4)],
            stream_dep_ms_mean_A_B=[round(float(np.concatenate(dep[:4]).mean()), 4), round(float(np.concatenate(dep[4:]).mean()), 4)],
            enqueue_to_submit_ms_mean_A_B=[round(float(np.concatenate(sub[:4]).mean()), 4), round(float(np.concatenate(sub[4:]).mean()), 4)],
            h2d_service_s_per_rank=[round(float(hsv[:, r].sum()), 3) for r in range(W)],
            exposed_wait_s_per_rank=[round(float(wait[:, r].sum()), 3) for r in range(W)],
            grouped_gemm_s_per_rank=[round(float(gemm[:, r].sum()), 3) for r in range(W)],
            sum_over_layers_max_rank_wait_plus_gemm_s=round(float(own.max(1).sum()), 3),
            sum_over_layers_mean_rank_wait_plus_gemm_s=round(float(own.mean(1).sum()), 3),
            critical_rank_share=[round(float((crit == r).mean()), 3) for r in range(W)],
            critical_group_share_A=round(float((crit < 4).mean()), 3),
            phase_totals_s_mean_A_B={p: [round(float(v[:4].mean()), 3), round(float(v[4:].mean()), 3)] for p, v in tot.items()},
            wall_seconds=[round(r['wall_seconds'], 3) for r in ranks],
            events=len(keys))
        report[label] = rep
    out = Path(__file__).resolve().parent / 'DIAG_ANALYSIS.json'
    out.write_text(json.dumps(report, indent=1) + '\n')
    for label, rep in report.items():
        print('=====', label)
        for k, v in rep.items():
            if k != 'phase_totals_s_mean_A_B':
                print(f'  {k}: {v}')
        print('  phase totals (A mean, B mean):')
        for p, v in sorted(rep['phase_totals_s_mean_A_B'].items(), key=lambda kv: -sum(kv[1])):
            print(f'    {p:45s} {v}')


if __name__ == '__main__':
    main()
