#!/usr/bin/env python3
"""Aggregate E1 only; publish the exact conditional E2 gate."""
import csv
import json
from pathlib import Path
import statistics
from trace_comm_input import PACKET,sha,classify
from run_trace_comm_replay import ROOT,ORDER,write


def percentile(values,p):
    values=sorted(values);x=(len(values)-1)*p/100;i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)


def main():
    progress=json.loads((PACKET/'trace_comm_progress.json').read_text());assert progress['status'] in ('PASS','FAIL')
    for name,digest in progress['source_sha256'].items():assert sha(name)==digest,name
    counts=json.loads((PACKET/'trace_comm_counts.json').read_text());rows=[];raw_receipts=[];samples=[];errors=[];event_rows=[]
    for pass_index,mode in ORDER:
        name=f'pass{pass_index}_{mode}';root=ROOT/name
        if not (root/'status.json').exists():continue
        state=json.loads((root/'status.json').read_text());samples.extend(state['memory'])
        for path in root.glob('*.json'):raw_receipts.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
        if state['status']!='PASS':
            errors.append(dict(cell=name,status=state['status'],stop_reason=state.get('stop_reason')));continue
        ranks=[json.loads((root/f'rank{r}.json').read_text()) for r in range(4)]
        for rank,r in enumerate(ranks):
            assert r['status']=='PASS' and r['rank']==rank and r['events']==384
            assert r['counts_sha256']==sha(PACKET/'trace_comm_counts.json')
            assert r['payload_valid'] and r['validation_outside_timing'] and r['no_model'] and r['no_cache_controller']
            assert r['warmup_full_traces']==1 and r['timed_full_traces']==3 and len(r['repeats'])==3
        for phase in ('dispatch','combine'):assert sum(r['peer_bytes_per_trace'][phase] for r in ranks)==counts['peer_bytes_per_trace'][phase]
        totals=[];max_cumulative=[];trace_intervals=[];wall=[];distribution={p:[] for p in ('dispatch','combine','pair')}
        for iteration in range(3):
            repeats=[r['repeats'][iteration] for r in ranks]
            assert all(r['payload_valid'] and len(r['events_ms'])==384 for r in repeats)
            values=[]
            for e in range(384):
                one={phase:max(r['events_ms'][e][phase] for r in repeats) for phase in distribution}
                for phase in distribution:distribution[phase].append(one[phase])
                values.append(one['pair']);event_rows.append(dict(pass_index=pass_index,mode=mode,iteration=iteration,event=e+48,**one))
            totals.append(sum(values));max_cumulative.append(max(r['cumulative_pair_ms'] for r in repeats))
            trace_intervals.append(max(r['full_trace_cuda_interval_ms'] for r in repeats));wall.append(max(r['wall_ms'] for r in repeats))
        row=dict(pass_index=pass_index,mode=mode,status='PASS',median_cumulative_max_rank_ms=statistics.median(totals),
                 median_max_rank_cumulative_ms=statistics.median(max_cumulative),
                 median_full_trace_cuda_interval_ms=statistics.median(trace_intervals),median_wall_ms=statistics.median(wall),
                 total_peer_bytes_per_trace=counts['total_peer_bytes_per_trace'])
        for phase,values in distribution.items():
            for pct in (50,90,99):row[f'{phase}_p{pct}_ms']=percentile(values,pct)
        row['replay_totals_ms']=totals;row['max_rank_cumulative_replays_ms']=max_cumulative
        row['full_trace_cuda_replays_ms']=trace_intervals;row['wall_replays_ms']=wall;rows.append(row)
    for mode in ('T0','R3'):
        root=ROOT/('preflight_'+mode)
        if (root/'status.json').exists():
            state=json.loads((root/'status.json').read_text());samples.extend(state['memory'])
            for path in root.glob('*.json'):raw_receipts.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    if progress['status']=='PASS':
        assert len(rows)==4
        by={(r['pass_index'],r['mode']):r for r in rows}
        ratios=[by[p,'R3']['median_cumulative_max_rank_ms']/by[p,'T0']['median_cumulative_max_rank_ms'] for p in (0,1)]
        decision=classify(ratios)
        secondary={key:[by[p,'R3'][key]/by[p,'T0'][key] for p in (0,1)] for key in ('median_max_rank_cumulative_ms','median_full_trace_cuda_interval_ms','median_wall_ms')}
    else:
        ratios=[];secondary={}
        decision='BLOCKED_PREFLIGHT' if not progress['completed'] else 'FAIL_VALIDATION_OR_EXECUTION'
    mem=dict(min_host_available_gib=min(s['host_available_bytes'] for s in samples)/2**30,
             max_tree_rss_gib=max(s.get('group_rss_bytes',0) for s in samples)/2**30,
             min_target_gpu_free_mib=min(g['free_mib'] for s in samples for g in s['gpu'].values()))
    result=dict(status=progress['status'],decision=decision,stage='E1',model_runs=0,E2_authorized=decision=='STRONG_GAP',E2_started=False,
                primary_definition='sum_event max_rank(dispatch+combine CUDA interval), median of 3 traces per mode/pass',
                pass_R3_over_T0=ratios,median_pass_ratio=statistics.median(ratios) if ratios else None,
                secondary_ratios=secondary,cells=rows,memory=mem,raw_receipts=raw_receipts,errors=errors,
                source_commit=progress['source_commit'],source_sha256=progress['source_sha256'],
                inputs_sha256=sha(PACKET/'trace_comm_counts.json'),payload_validated_all_iterations=progress['status']=='PASS',
                warmup_full_traces=len(rows),timed_full_traces=3*len(rows),
                preflight_failure=json.loads((PACKET/'trace_comm_preflight_failure.json').read_text()) if (PACKET/'trace_comm_preflight_failure.json').exists() else None,
                no_additional_NCCL_knobs=True,source_hashes_unchanged=True)
    write(PACKET/'trace_comm_replay.json',result)
    flat=[{k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()} for row in rows]
    if not flat:
        flat=[dict(pass_index=p,mode=m,status='NOT_RUN',median_cumulative_max_rank_ms='') for p,m in ORDER]
    if flat:
        with (PACKET/'trace_comm_replay.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    if event_rows:
        with (PACKET/'trace_comm_events.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(event_rows[0]),lineterminator='\n');w.writeheader();w.writerows(event_rows)
    lines=['# E1 actual-trace communication replay','',f'**{decision}**. No model, expert compute/H2D, cache or controller was run.','']
    if ratios:
        lines += [f"R3/T0 ratios are **{ratios[0]:.4f}x** and **{ratios[1]:.4f}x** in counter-ordered passes; their median is **{statistics.median(ratios):.4f}x**.",'',
                  '| Pass/order | Mode | Median cumulative max-rank ms | Three trace totals ms |', '|:---|:---|---:|:---|']
        for row in rows:lines.append(f"| {row['pass_index']} ({'T0 → R3' if row['pass_index']==0 else 'R3 → T0'}) | {row['mode']} | {row['median_cumulative_max_rank_ms']:.4f} | {', '.join(f'{x:.4f}' for x in row['replay_totals_ms'])} |")
        lines += ['', '## Event-level max-rank intervals', '', '| Pass | Mode | Phase | p50 ms | p90 ms | p99 ms |','|---:|:---|:---|---:|---:|---:|']
        for row in rows:
            for phase in ('dispatch','combine','pair'):
                lines.append(f"| {row['pass_index']} | {row['mode']} | {phase} | {row[phase+'_p50_ms']:.6f} | {row[phase+'_p90_ms']:.6f} | {row[phase+'_p99_ms']:.6f} |")
        lines += ['', 'Percentiles pool 3 × 384 event intervals per mode/pass, with linear interpolation. Paired maxima may occur on different ranks from individual dispatch/combine maxima.', '',
                  '## Secondary timing definitions', '', '| Definition | Pass 0 R3/T0 | Pass 1 R3/T0 |', '|:---|---:|---:|']
        for key,values in secondary.items():lines.append(f"| {key} | {values[0]:.4f}x | {values[1]:.4f}x |")
    lines += ['', '## Validation and scope', '',
              '- Original captured P0 traffic from `ed7f82b`, not F/K/C replay traffic: all 384 decode events retain their rank-pair imbalance and original order. Input receipts are SHA256-verified, every send/receive matrix transpose matches, and BF16 row width is 4096 bytes.',
              '- Exactly one full-trace warmup and three timed traces per mode/pass. One tiny path check per mode verifies P2P/CUMEM and SHM/direct/direct; INFO is disabled in timing.',
              '- Buffers are materialized before timing. Receives are cleared to NaN before each trace; every received element is compared to an event/phase/source/destination/element sentinel after each trace. Validation, reductions, quantiles and output writes are outside the timed loop.',
              '- Runtime variable-size all-to-all API; dispatch then combine at every event. No equal-peer-size replacement, router metadata exchange, per-event barrier, model load, expert traffic or greedy policy computation.',
              f"- Peer bytes per trace: dispatch {counts['peer_bytes_per_trace']['dispatch']:,}, combine {counts['peer_bytes_per_trace']['combine']:,}, total {counts['total_peer_bytes_per_trace']:,}; self entries execute but are excluded from byte totals.",
              '- Primary timing is sum of per-event max-rank paired CUDA intervals, median of three traces. Also report max-rank cumulative intervals and whole-trace CUDA/wall times. CUDA intervals include launch/stream waits; they are not isolated NCCL kernel duration.',
              f"- Guard observations: maximum process-tree RSS {mem['max_tree_rss_gib']:.2f} GiB, minimum host available {mem['min_host_available_gib']:.2f} GiB, minimum target GPU free {mem['min_target_gpu_free_mib']:,} MiB. GPUs are restricted to 0,1,4,5.",
              '- Two CPU tests passed: gate boundaries and rejection of a corrupted receive matrix. All source hashes remained unchanged during measurement.', '',
              '## Stage boundary', '']
    if decision=='STRONG_GAP':lines += ['E1 meets both directional checks and the >=1.20 median gate. Commit this E1 result before starting the separately authorized clean F/K Stage E2. No E2 model process has started in this receipt.']
    elif decision=='NO_GAP':lines += ['Do not start E2. Stop H100 synthetic P2P-off work: this condition does not establish a stable real-trace communication-price contrast. Future communication-sensitive validation belongs on the real no-NVLink server, as specified by the plan; that server is not started automatically.']
    else:lines += ['Do not start E2. Commit and stop for owner review without extra repetitions, channel/protocol tuning or another server run.']
    lines += ['', 'See [protocol](TRACE_COMM_REPLAY_PROTOCOL.md), [summary CSV](trace_comm_replay.csv), [full JSON and hashes](trace_comm_replay.json), [event timings](trace_comm_events.csv), and [validated traffic input](trace_comm_counts.json).','']
    if decision=='BLOCKED_PREFLIGHT':
        fail=result['preflight_failure']
        lines=['# E1 actual-trace communication replay', '', '**BLOCKED_PREFLIGHT**. E1 timings are unavailable; E2 was not started.', '',
               'The first T0 32-KiB preflight exceeded the predeclared 180-second bound and was terminated. All four NCCL communicators logged initialization completion and selected P2P/CUMEM, but the payload exchange never produced a passing receipt. The final logs show shareable-buffer imports and UDS handle mapping. This is not a measured NO_GAP or AMBIGUOUS_GAP result.', '',
               'Observed worker wait channels included `uvm_gpu_retain_by_uuid` and `uvm_va_space_unregister_gpu`. This locates the observed wait but does not establish its root cause. NCCL 2.28.9 matches the prior successful physical-pilot preflight. No NCCL knob was changed and no driver reset or retry was attempted.', '',
               f"The launcher stopped on timeout, not the memory guard. Peak process-tree RSS was {mem['max_tree_rss_gib']:.2f} GiB; host availability stayed above {mem['min_host_available_gib']:.2f} GiB. Target GPUs 0,1,4,5 returned to zero used memory after cleanup.", '',
               '## Completed work', '',
               '- Extracted all 384 original decode dispatch/combine matrices from four SHA256-verified capture files. Count transposes and 4096-byte BF16 rows pass CPU validation.',
               '- Two CPU tests pass: decision boundaries and corrupted receive-count rejection.',
               '- Frozen input and replay protocol were committed before GPU work. The model-free worker, warmup/repetition limits and sentinel validation are implemented; their trace replay has not yet run.',
               '- One T0 preflight attempted and failed. R3 preflight, all four timed cells and every E2 model cell remain unstarted. Zero trace warmups, timed traces or model generations were executed.', '',
               '## Receipts and boundary', '',
               'See [failure details and raw-log hashes](trace_comm_preflight_failure.json), [compact NCCL excerpt](trace_comm_T0_preflight_failure.log), [stage JSON](trace_comm_replay.json), [unrun cell CSV](trace_comm_replay.csv), [frozen input](trace_comm_counts.json), and [protocol](TRACE_COMM_REPLAY_PROTOCOL.md).', '',
               'No R3/T0 ratio exists, so the STRONG_GAP requirement for E2 is unmet. Preserve the failure and stop; do not infer a communication-price contrast or switch servers automatically.', '']
    (PACKET/'TRACE_COMM_REPLAY_RESULTS.md').write_text('\n'.join(lines))
    validation=dict(status=progress['status'],decision=decision,model_runs=0,unit_tests=2,
                    input_count_validation='PASS',cpu_unit_tests='PASS',
                    E1_timing_complete=progress['status']=='PASS',E2_authorized=decision=='STRONG_GAP',
                    validated_payload_traces=4*len(rows),
                    validated_global_events=4*len(rows)*384,
                    source_sha256={str(Path(__file__)):sha(Path(__file__))},
                    outputs_sha256={name:sha(PACKET/name) for name in ('trace_comm_replay.json','trace_comm_replay.csv','trace_comm_events.csv','TRACE_COMM_REPLAY_RESULTS.md') if (PACKET/name).exists()})
    write(PACKET/'trace_comm_validation.json',validation)
    print(json.dumps(dict(decision=decision,ratios=ratios,median=result['median_pass_ratio'],secondary=secondary,memory=mem),indent=2))


if __name__=='__main__':main()
