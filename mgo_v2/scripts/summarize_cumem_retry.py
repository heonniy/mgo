#!/usr/bin/env python3
"""Persist bounded D1/D2 diagnosis; no GPU launches."""
import csv
import json
from pathlib import Path
from trace_comm_input import PACKET,sha
from run_cumem_preflight_retry import ROOT,write


def main():
    progress=json.loads((PACKET/'cumem_retry_progress.json').read_text())
    assert progress['status'] in ('COMPLETE','BLOCKED')
    for path,digest in progress['source_sha256'].items():assert sha(path)==digest
    trials=progress['trials'];rows=[];raw=[]
    for t in trials:
        rows.append(dict(trial=t['trial'],cumem_disabled=t['cumem_disabled'],status=t['status'],
                         elapsed_seconds=t['finished_unix']-t['started_unix'],signal_elapsed_seconds=t.get('signal_elapsed_seconds'),
                         payload_valid_all_ranks=t['payload_valid_all_ranks'],selected_transports=';'.join(t['selected_transports']),
                         target_gpus_released=t['target_gpus_released'],stop_reason=t['stop_reason']))
    for path in sorted(ROOT.glob('*/*')):
        if path.is_file():raw.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    diagnosis=progress.get('diagnosis','BLOCKED_DIAGNOSIS')
    if progress['status']=='COMPLETE':
        assert len(trials)==4 and [t['cumem_disabled'] for t in trials]==[False,False,False,True]
        assert all(t['timeout_seconds']==90 for t in trials)
        for t in trials:
            if t['status']=='PASS':
                rs=[json.loads((ROOT/t['trial']/f'rank{r}.json').read_text()) for r in range(4)]
                assert all(r['payload_valid'] and r['rank']==i for i,r in enumerate(rs))
            elif t['status']=='TIMEOUT':assert t['signal_elapsed_seconds']<90.1
    samples=[s for t in trials for s in t['memory']]
    mem=dict(max_tree_rss_gib=max((s.get('group_rss_bytes',0) for s in samples),default=0)/2**30,
             min_host_available_gib=min((s['host_available_bytes'] for s in samples),default=0)/2**30,
             min_target_gpu_free_mib=min((g['free_mib'] for s in samples for g in s['gpu'].values()),default=0))
    result=dict(status=progress['status'],diagnosis=diagnosis,plan_commit=progress['plan_commit'],trials=rows,
                default_passes=sum(t['status']=='PASS' for t in trials[:3]),default_trials=3,
                E1_retry_authorized=progress.get('E1_retry_authorized',False),E1_started=False,model_runs=0,
                memory=mem,source_sha256=progress['source_sha256'],raw_receipts=raw,
                other_gpu_jobs_modified=False,driver_reset=False,transport_knob_search=False,
                cumem_disabled_baseline_adopted=False)
    write(PACKET/'cumem_preflight_retry.json',result)
    with (PACKET/'cumem_preflight_retry.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['trial','status'],lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['# Bounded P2P/cuMem preflight diagnosis','',f"**{diagnosis}**. Default T0 payload validation passed {result['default_passes']}/3 fresh trials.",'',
           '| Trial | cuMem override | Outcome | Selected path(s) | Wall s including cleanup |', '|:---|:---|:---|:---|---:|']
    for r in rows:lines.append(f"| {r['trial']} | {'NCCL_CUMEM_ENABLE=0' if r['cumem_disabled'] else 'unset'} | {r['status']} | {r['selected_transports']} | {r['elapsed_seconds']:.3f} |")
    lines+=['','Each trial sends exactly 32 KiB per peer, uses a fresh four-rank process group on GPUs 0,1,4,5, and has a 90-second active-runtime limit. Wall time above includes termination/cleanup after a timeout; timeouts are failures, not performance samples. PASS requires all four payload receipts; default trials additionally require P2P/CUMEM. The alternate path is reported as observed.', '',
            '## Evidence and resource isolation','',
            '- Each trial has a prelaunch snapshot of all eight GPU process/utilization states, target free memory, host availability, ambient/launch NCCL environment, and CUDA/PyTorch/NCCL versions. Snapshots are committed as `cumem_<trial>_environment.json`.',
            '- INFO logging is confined to these four diagnostics. Compact per-rank excerpts retain initialization/path evidence and final 50 lines; raw logs and receipts have hashes in the JSON.',
            '- Timeout evidence records worker wait channels, the last process/GPU/host memory sample and the last 50 NCCL lines per rank before terminating only the trial process group.',
            f"- Peak trial process-tree RSS {mem['max_tree_rss_gib']:.3f} GiB; minimum host available {mem['min_host_available_gib']:.2f} GiB; minimum target GPU free {mem['min_target_gpu_free_mib']:,} MiB.",
            '- No driver reset, server reboot, channel/protocol tuning, model generation or changes to workloads on GPUs 2,3,6,7. The cuMem-disabled trial is diagnostic only.', '', '## Decision boundary','']
    if diagnosis=='TRANSIENT_RECOVERED':lines+=['All three default trials passed on P2P/CUMEM. Treat the prior timeout as an isolated/transient infrastructure stall for this study, without claiming an exact root cause. Commit this diagnosis first, then re-run the unchanged E1 trace measurement on its original T0/R3 environments in a fresh raw-result root.']
    elif diagnosis=='ALTERNATE_PATH_FAILURE':lines+=['The default path passed 3/3; the alternate-path failure does not block the original E1 retry. Commit this diagnosis first. Do not adopt cuMem-disabled T0.']
    elif diagnosis=='CUMEM_PATH_UNSTABLE':lines+=['At least one default trial failed while the cuMem-disabled diagnostic passed. This supports cuMem-path-specific instability, without establishing the driver root cause. Stop after diagnosis: no E1 retry, no alternate experimental baseline, and no model work.']
    elif diagnosis=='BROADER_P2P_UVM_FAILURE':lines+=['Both the default group and the alternate diagnostic failed their bounded checks. Stop after diagnosis; do not infer a cuMem-only root cause, search more NCCL knobs, or start E1/model work.']
    else:lines+=['Diagnosis did not complete; preserve receipts and stop without further GPU launches.']
    lines+=['','See [CSV](cumem_preflight_retry.csv), [JSON/provenance](cumem_preflight_retry.json), and [owner plan](CUMEM_PREFLIGHT_RETRY.md).','']
    (PACKET/'CUMEM_PREFLIGHT_RETRY_RESULTS.md').write_text('\n'.join(lines))
    write(PACKET/'cumem_preflight_validation.json',dict(status=progress['status'],diagnosis=diagnosis,
          completed_trials=len(trials),source_hashes_unchanged=True,per_trial_timeout_seconds=90,
          source_sha256={str(Path(__file__)):sha(Path(__file__))},
          output_sha256={n:sha(PACKET/n) for n in ('cumem_preflight_retry.csv','cumem_preflight_retry.json','CUMEM_PREFLIGHT_RETRY_RESULTS.md')}))
    print(json.dumps(dict(diagnosis=diagnosis,default_passes=result['default_passes'],E1_retry_authorized=result['E1_retry_authorized'],memory=mem),indent=2))


if __name__=='__main__':main()
