#!/usr/bin/env python3
"""Apply unchanged E1 timing conventions/gate to matched IPC/SHM receipts."""
import csv
import json
from pathlib import Path
import statistics
from trace_comm_input import PACKET,sha,classify
from run_ipc_baseline_rebase import ROOT,ORDER,LABELS,write
from summarize_trace_comm_replay import percentile


def main():
    progress=json.loads((PACKET/'ipc_rebase_progress.json').read_text());assert progress['status'] in ('PASS','FAIL')
    for name,digest in progress['source_sha256'].items():assert sha(name)==digest,name
    assert all(sha(PACKET/n)==digest for n,digest in progress['cpu_result_sha256'].items())
    counts=json.loads((PACKET/'trace_comm_counts.json').read_text())
    rows=[];events=[];receipts=[];samples=[];preflights=[]
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir():continue
        if (folder/'status.json').exists():samples+=json.loads((folder/'status.json').read_text())['memory']
        for path in folder.iterdir():
            if path.is_file():receipts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    for mode in ('T0','R3'):
        folder=ROOT/('preflight_'+mode)
        if (folder/'status.json').exists():
            r=json.loads((folder/'status.json').read_text())
            preflights.append(dict(mode=LABELS[mode],process_status=r['status'],acceptance_passed=mode in progress['preflights'],
                                   selected_transports=r.get('selected_transports'),one_node_four_local_ranks=r.get('one_node_four_local_ranks'),
                                   transport_env=r['transport_env']))
    for pass_index,mode in ORDER:
        name=f'pass{pass_index}_{mode}'
        if name not in progress['completed']:continue
        rs=[json.loads((ROOT/name/f'rank{r}.json').read_text()) for r in range(4)]
        expected={'NCCL_CUMEM_ENABLE':'0'}
        if mode=='R3':expected.update(NCCL_P2P_LEVEL='LOC',NCCL_IB_DISABLE='1')
        for rank,r in enumerate(rs):
            assert r['status']=='PASS' and r['rank']==rank and r['ipc_rebase'] and r['transport_env']==expected
            assert r['counts_sha256']==sha(PACKET/'trace_comm_counts.json') and r['events']==384
            assert r['payload_valid'] and r['validation_outside_timing'] and r['no_model'] and r['no_cache_controller']
            assert r['warmup_full_traces']==1 and r['timed_full_traces']==3 and len(r['repeats'])==3
            assert not r['nccl_info_in_timing']
        for phase in ('dispatch','combine'):assert sum(r['peer_bytes_per_trace'][phase] for r in rs)==counts['peer_bytes_per_trace'][phase]
        totals=[];max_cumulative=[];whole=[];wall=[];distribution={p:[] for p in ('dispatch','combine','pair')}
        for iteration in range(3):
            repetitions=[r['repeats'][iteration] for r in rs]
            assert all(r['payload_valid'] and len(r['events_ms'])==384 for r in repetitions)
            total=0
            for e in range(384):
                interval={phase:max(r['events_ms'][e][phase] for r in repetitions) for phase in distribution}
                total+=interval['pair']
                for phase in distribution:distribution[phase].append(interval[phase])
                events.append(dict(pass_index=pass_index,mode=LABELS[mode],iteration=iteration,event=e+48,**interval))
            totals.append(total);max_cumulative.append(max(r['cumulative_pair_ms'] for r in repetitions))
            whole.append(max(r['full_trace_cuda_interval_ms'] for r in repetitions));wall.append(max(r['wall_ms'] for r in repetitions))
        row=dict(pass_index=pass_index,mode=mode,label=LABELS[mode],status='PASS',median_cumulative_max_rank_ms=statistics.median(totals),
                 median_max_rank_cumulative_ms=statistics.median(max_cumulative),median_full_trace_cuda_interval_ms=statistics.median(whole),
                 median_wall_ms=statistics.median(wall),peer_bytes_per_trace=counts['total_peer_bytes_per_trace'],
                 cumulative_trace_samples_ms=totals,max_rank_cumulative_samples_ms=max_cumulative,whole_trace_samples_ms=whole,wall_samples_ms=wall)
        for phase,values in distribution.items():
            for pct in (50,90,99):row[f'{phase}_p{pct}_ms']=percentile(values,pct)
        rows.append(row)
    ratios=[];secondary={}
    if progress['status']=='PASS':
        assert len(rows)==4 and len(preflights)==2 and all(p['acceptance_passed'] for p in preflights)
        by={(r['pass_index'],r['mode']):r for r in rows}
        ratios=[by[p,'R3']['median_cumulative_max_rank_ms']/by[p,'T0']['median_cumulative_max_rank_ms'] for p in (0,1)]
        decision=classify(ratios)
        secondary={key:[by[p,'R3'][key]/by[p,'T0'][key] for p in (0,1)] for key in ('median_max_rank_cumulative_ms','median_full_trace_cuda_interval_ms','median_wall_ms')}
    else:decision='IPC_REBASE_BLOCKED' if len(progress['preflights'])<2 else 'IPC_TRACE_REPLAY_FAILED'
    mem=dict(min_host_available_gib=min(s['host_available_bytes'] for s in samples)/2**30,
             max_tree_rss_gib=max(s.get('group_rss_bytes',0) for s in samples)/2**30,
             min_target_gpu_free_mib=min(g['free_mib'] for s in samples for g in s['gpu'].values()))
    result=dict(status=progress['status'],decision=decision,plan_commit=progress['plan_commit'],labels=LABELS,
                preflights=preflights,cells=rows,pass_R3_over_T0=ratios,median_pass_ratio=statistics.median(ratios) if ratios else None,
                secondary_ratios=secondary,memory=mem,source_commit=progress['source_commit'],source_sha256=progress['source_sha256'],
                cpu_screen_rerun=False,cpu_results_unchanged=True,cpu_result_sha256=progress['cpu_result_sha256'],
                source_trace='ed7f82b original captured decode matrices',counts_sha256=sha(PACKET/'trace_comm_counts.json'),
                model_runs=0,I2_authorized=decision=='STRONG_GAP',I2_started=False,raw_receipts=receipts,
                error=progress.get('error'),warmup_traces=len(rows),timed_traces=len(rows)*3)
    write(PACKET/'ipc_trace_comm_replay.json',result)
    flat=[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows]
    if not flat:flat=[dict(pass_index=p,label=LABELS[m],status='NOT_RUN') for p,m in ORDER]
    with (PACKET/'ipc_trace_comm_replay.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    if events:
        with (PACKET/'ipc_trace_comm_events.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(events[0]),lineterminator='\n');w.writeheader();w.writerows(events)
    lines=['# IPC baseline rebase and actual-trace communication gate','',f'**{decision}**. CPU Pareto points, F/K selection and frozen schedule metadata are unchanged (SHA256 verified); no CPU screen or model was run.','',
           '## I0 paired transport acceptance','', '| Mode | Acceptance | Observed channel paths |','|:---|:---|:---|']
    for p in preflights:lines.append(f"| {p['mode']} | {'PASS' if p['acceptance_passed'] else 'FAIL'} | {p['selected_transports']} |")
    lines+=['','Both modes set `NCCL_CUMEM_ENABLE=0`. T0-IPC clears other transport overrides; R3-SHM additionally sets P2P_LEVEL=LOC and IB_DISABLE=1. Accepted smokes require one node/four local ranks and all-rank payload validation. T0 permits only P2P/IPC; R3 permits only SHM with no P2P/NET. INFO is confined to I0.','']
    if ratios:
        lines += [f"I1 R3-SHM/T0-IPC ratios: **{ratios[0]:.4f}x** in pass 0 and **{ratios[1]:.4f}x** in pass 1; median **{statistics.median(ratios):.4f}x**.",'',
                  '| Pass/order | Mode | Median cumulative max-rank ms | Three trace totals ms |','|:---|:---|---:|:---|']
        for r in rows:lines.append(f"| {r['pass_index']} ({'T0 → R3' if r['pass_index']==0 else 'R3 → T0'}) | {r['label']} | {r['median_cumulative_max_rank_ms']:.4f} | {', '.join(f'{v:.4f}' for v in r['cumulative_trace_samples_ms'])} |")
        lines+=['','## Event-level max-rank percentiles','', '| Pass | Mode | Phase | p50 ms | p90 ms | p99 ms |','|---:|:---|:---|---:|---:|---:|']
        for r in rows:
            for phase in ('dispatch','combine','pair'):lines.append(f"| {r['pass_index']} | {r['label']} | {phase} | {r[phase+'_p50_ms']:.6f} | {r[phase+'_p90_ms']:.6f} | {r[phase+'_p99_ms']:.6f} |")
        lines+=['','## Secondary timing definitions','', '| Definition | Pass 0 R3/T0 | Pass 1 R3/T0 |','|:---|---:|---:|']
        for key,rs in secondary.items():lines.append(f"| {key} | {rs[0]:.4f}x | {rs[1]:.4f}x |")
    else:lines += [f"Stopped without a complete ratio: `{progress.get('error')}`. No clean F/K model run is authorized.",'']
    lines+=['','## Validation and measurement scope','',
            '- Original captured 384-event variable-size decode sequence, including rank-pair imbalance and original dispatch/combine order. Inputs and worker sources remain hash-identical during this run.',
            '- One untimed full-trace warmup and three timed traces per mode/pass. All buffers are materialized outside timing, receives reset to NaN, and every payload element validated after each trace against its deterministic sentinel.',
            '- The existing timing loop and aggregation are unchanged: primary = sum_event max_rank(dispatch+combine CUDA interval), median of three totals; gate = median of two pass ratios. Event percentiles pool 1,152 max-rank intervals per mode/pass. No router metadata, expert H2D/compute or cache/controller work.',
            f"- Peer bytes per trace: dispatch {counts['peer_bytes_per_trace']['dispatch']:,}, combine {counts['peer_bytes_per_trace']['combine']:,}, total {counts['total_peer_bytes_per_trace']:,}. Self entries execute but are excluded from byte totals.",
            '- CUDA intervals include API/launch/stream waits and are not isolated kernel durations. Whole-trace and reversed rank-max aggregation are published as secondary definitions; no threshold or timing definition was changed after observing results.',
            f"- Peak process-tree RSS {mem['max_tree_rss_gib']:.3f} GiB; minimum host available {mem['min_host_available_gib']:.2f} GiB; minimum target GPU free {mem['min_target_gpu_free_mib']:,} MiB. No model run or CPU Pareto rerun.",
            '- R3 is a software-disabled P2P condition on the same NVSwitch host; it is not a physically no-NVLink server. The prior cuMem physical pilot remains historical.', '', '## Stage boundary','']
    if decision=='STRONG_GAP':lines+=['R3 is slower in both passes and the median meets 1.20x. Commit I0/I1 first; only then may the separately validated clean F/K stage use these matched IPC/SHM environments.']
    elif decision=='NO_GAP':lines+=['Do not start clean F/K. The synthetic condition does not establish a stable real-trace communication-price contrast. Stop without additional repeats, rho values, channel/protocol tuning or automatic migration to another server.']
    else:lines+=['Commit and stop for owner review. Clean F/K is not authorized by this result; do not add repetitions or tune transport.']
    lines+=['','See [CSV](ipc_trace_comm_replay.csv), [full result and provenance](ipc_trace_comm_replay.json), [validation](ipc_baseline_validation.json), and [execution conventions](IPC_REBASE_EXECUTION.md).','']
    (PACKET/'IPC_BASELINE_REBASE_RESULTS.md').write_text('\n'.join(lines))
    write(PACKET/'ipc_baseline_validation.json',dict(status=progress['status'],decision=decision,cpu_results_unchanged=True,cpu_screen_rerun=False,
          sources_unchanged=True,validated_trace_replays=len(rows)*4,validated_global_events=len(rows)*4*384,
          model_runs=0,I2_authorized=decision=='STRONG_GAP',source_sha256={str(Path(__file__)):sha(Path(__file__))},
          output_sha256={n:sha(PACKET/n) for n in ('ipc_trace_comm_replay.csv','ipc_trace_comm_replay.json','ipc_trace_comm_events.csv','IPC_BASELINE_REBASE_RESULTS.md') if (PACKET/n).exists()}))
    print(json.dumps(dict(decision=decision,ratios=ratios,median=result['median_pass_ratio'],secondary=secondary,memory=mem),indent=2))


if __name__=='__main__':main()
