"""Harvest existing LiveMetadata.collect_profiled phase spans from receipts (CPU only).

Reads generation_diagnostic_rank*.json written by scripts/generation_phase_diagnostics.py
(post-generation diagnostic pass of examples/headline_ours_worker.py). No new measurement.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

R = Path('/home/hwlee/mgo-results')
RUNS = [  # (dir, local_batch, variant)
 ('grouped_policy_scaling_20261009/jobs/r4_a_la_ca_near_8_diagnostic_v1', 8, 'main_config_serial'),
 ('grouped_policy_scaling_20261009/jobs/r4_a_la_ca_near_16_diagnostic_v1', 16, 'main_config_serial'),
 ('expert_grouping_ablation_20261009/jobs/r4_a_diagnostic_v1', 16, 'main_config_serial'),
 ('expert_grouping_ablation_20261009/jobs/r4_a_diagnostic_v4', 16, 'main_config_serial'),
 ('ep_overhead_r4_b64_20261009/jobs/r4_a_diagnostic_v1', 64, 'main_config_serial'),
 ('ep_overhead_r4_b64_20261009/jobs/r4_a_diagnostic_v3', 64, 'main_config_serial'),
 ('expert_grouping_ablation_20261009/jobs/r4_a_diagnostic_v5', 16, 'serial+compiled_dense'),
 ('grouped_policy_scaling_20261009/jobs/r4_n_la_ca_near_8_diagnostic_v1', 8, 'grouped_hit_then_miss+compiled_dense'),
 ('grouped_policy_scaling_20261009/jobs/r4_n_la_ca_near_16_diagnostic_v1', 16, 'grouped_hit_then_miss+compiled_dense'),
 ('grouped_policy_scaling_20261009/jobs/r4_n_la_ca_near_64_diagnostic_v1', 64, 'grouped_hit_then_miss+compiled_dense'),
]
PH = ['metadata_gpu_packet_pack', 'metadata_rank_all_gather', 'metadata_device_to_host',
      'metadata_cpu_parse', 'metadata_gate_history', 'metadata_demand_histogram']
GLUE = 'moe.metadata'  # residual between sub-phase marks (instrumentation glue)

def q(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')

def main(out):
    rows = []
    for d, B, variant in RUNS:
        st = json.loads((R / d / 'status.json').read_text())
        cmd = st['command']; assert 'LA_CA_NEAR' in cmd and 'R4_C30' in cmd[cmd.index('--cell') + 1]
        main_tpot = [json.loads((R / d / f'repeat{i}.json').read_text()).get('TPOT') for i in (1,) if (R / d / f'repeat{i}.json').exists()]
        for rank in range(4):
            g = json.loads((R / d / f'generation_diagnostic_rank{rank}.json').read_text())
            per_event = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))  # event -> phase -> [stream_s, wall_s]
            step_of = {}
            for s in g['segments']:
                ev = s.get('event_index'); step = s.get('step')
                if ev is None or ev < 48 or s['phase'] not in PH + [GLUE]: continue
                per_event[ev][s['phase']][0] += s['stream_seconds']
                per_event[ev][s['phase']][1] += s.get('wall_ns', 0) / 1e9
                step_of[ev] = step
            evs = sorted(per_event)
            ready = np.asarray(g['output']['token_ready_ns'], dtype=np.int64)
            step_dt = np.diff(ready) / 1e9  # decode step s (1..63) latency incl. diagnostic overhead
            diag_tpot = float(np.median(step_dt))
            for ph in PH + [GLUE, 'metadata_total']:
                if ph == 'metadata_total':
                    st_v = np.array([sum(v[0] for v in per_event[e].values()) for e in evs])
                    wl_v = np.array([sum(v[1] for v in per_event[e].values()) for e in evs])
                else:
                    st_v = np.array([per_event[e][ph][0] for e in evs])
                    wl_v = np.array([per_event[e][ph][1] for e in evs])
                # per decode step (sum over 48 layer events) and fraction of that step's latency
                per_step = defaultdict(float)
                for e, v in zip(evs, st_v): per_step[step_of[e]] += v
                steps = sorted(k for k in per_step if 1 <= k <= len(step_dt))
                frac = np.array([per_step[k] / step_dt[k - 1] for k in steps])
                rows.append(dict(run=d, variant=variant, model='Qwen3-30B-A3B', R=4, local_batch=B,
                    rank=rank, phase=ph, n_events=len(evs),
                    stream_median_us=round(q(st_v, 50) * 1e6, 2), stream_p95_us=round(q(st_v, 95) * 1e6, 2),
                    wall_median_us=round(q(wl_v, 50) * 1e6, 2), wall_p95_us=round(q(wl_v, 95) * 1e6, 2),
                    per_step_median_ms=round(q(np.array([per_step[k] for k in steps]), 50) * 1e3, 3),
                    frac_of_step_median=round(q(frac, 50), 4), frac_of_step_p95=round(q(frac, 95), 4),
                    diag_tpot_median_s=round(diag_tpot, 4),
                    job_target_tpot_s=round(main_tpot[0], 4) if main_tpot and main_tpot[0] else ''))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), 'rows ->', out)

if __name__ == '__main__': main(sys.argv[1])


def summary(out):
    """Per run: cross-rank view. all_gather min-over-ranks per event approximates
    collective latency with least peer wait; non-collective = pack+D2H+parse+gate+hist+glue."""
    rows = []
    for d, B, variant in RUNS:
        ag = {}; nonc = {}; tot = {}; steps = {}
        for rank in range(4):
            g = json.loads((R / d / f'generation_diagnostic_rank{rank}.json').read_text())
            a = defaultdict(float); n = defaultdict(float); t = defaultdict(float)
            for s in g['segments']:
                ev = s.get('event_index')
                if ev is None or ev < 48 or s['phase'] not in PH + [GLUE]: continue
                t[ev] += s['stream_seconds']
                if s['phase'] == 'metadata_rank_all_gather': a[ev] += s['stream_seconds']
                else: n[ev] += s['stream_seconds']
            ag[rank] = a; nonc[rank] = n; tot[rank] = t
            steps[rank] = np.diff(np.asarray(g['output']['token_ready_ns'], dtype=np.int64)) / 1e9
        evs = sorted(set.intersection(*[set(ag[r]) for r in range(4)]))
        agmin = np.array([min(ag[r][e] for r in range(4)) for e in evs])
        agmax = np.array([max(ag[r][e] for r in range(4)) for e in evs])
        nonc_all = np.concatenate([[nonc[r][e] for e in evs] for r in range(4)])
        nonc_max = np.array([max(nonc[r][e] for r in range(4)) for e in evs])
        step_med = float(np.median(np.concatenate([steps[r] for r in range(4)])))
        # lower-bound per step = sum_e (min-rank all-gather + max-rank non-collective)
        lb_ev = agmin + nonc_max
        per_step_lb = defaultdict(float)
        for e, v in zip(evs, lb_ev): per_step_lb[e // 48] += v
        lbs = np.array([per_step_lb[k] for k in sorted(per_step_lb)])
        rows.append(dict(run=d, variant=variant, local_batch=B, n_events=len(evs),
            allgather_min_rank_median_us=round(q(agmin, 50) * 1e6, 1), allgather_min_rank_p95_us=round(q(agmin, 95) * 1e6, 1),
            allgather_max_rank_median_us=round(q(agmax, 50) * 1e6, 1), allgather_max_rank_p95_us=round(q(agmax, 95) * 1e6, 1),
            noncollective_median_us=round(q(nonc_all, 50) * 1e6, 1), noncollective_p95_us=round(q(nonc_all, 95) * 1e6, 1),
            lower_bound_per_step_ms=round(q(lbs, 50) * 1e3, 2), diag_step_median_ms=round(step_med * 1e3, 1),
            lower_bound_frac_of_step=round(q(lbs, 50) / step_med, 4)))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows: print(r)

if __name__ == '__main__' and len(sys.argv) > 2: summary(sys.argv[2])
