"""Report primary Stage-I evidence and separate intrusive CPU/kernel diagnostics."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def union(intervals):
    result=0.;end=-float('inf')
    for left,right in sorted(intervals):
        if right>end:result+=right-max(left,end);end=right
    return result


def kernel_report(path):
    events=json.loads(path.read_text())['traceEvents']
    cpu=[e for e in events if e.get('cat')=='user_annotation' and e.get('ph')=='X' and e.get('name')=='pcie.executor_call']
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
    parents_all=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='pcie.moe_runtime' and e.get('ph')=='X']
    launches=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('ph')=='X']
    assert len(cpu)==48
    per_layer=[];families=collections.defaultdict(lambda:dict(count=0,kernel_sum_us=0.))
    for span in cpu:
        left=span['ts'];right=left+span['dur']
        # The executor is asynchronous; account for its final kernels using
        # the next enclosing moe_runtime end, where compute has synchronized.
        parents=[e for e in parents_all if e['ts']<=left and e['ts']+e['dur']>=right]
        assert len(parents)==1
        completed=parents[0]['ts']+parents[0]['dur']
        # Kernel CPU correlations assign operations to the executor precisely,
        # rather than accidentally including return/metadata kernels.
        children=[e for e in launches if left<=e['ts'] and e['ts']+e['dur']<=right]
        correlations={e.get('args',{}).get('correlation') for e in children};correlations.discard(None)
        local=[k for k in kernels if k.get('args',{}).get('correlation') in correlations]
        assert all(k['ts']+k['dur']<=completed+1 for k in local)
        windows=[(k['ts'],k['ts']+k['dur']) for k in local]
        for k in local:
            name=k['name'].lower()
            family='gemm' if 'gemm' in name else 'activation' if 'silu_product' in name else 'gather' if 'index' in name else 'weight_or_other'
            families[family]['count']+=1;families[family]['kernel_sum_us']+=k['dur']
        per_layer.append(dict(cpu_executor_us=span['dur'],kernel_count=len(local),kernel_active_union_us=union(windows),
                              gemm_count=sum('gemm' in k['name'].lower() for k in local),
                              kernel_stream_window_us=max((b for _,b in windows),default=left)-min((a for a,_ in windows),default=left)))
    return dict(scope='One early target decode forward under intrusive Torch CPU+CUDA profiler; not primary timing',
                layers=per_layer,families=dict(families),cpu_executor_us=sum(r['cpu_executor_us'] for r in per_layer),
                kernel_active_union_us=sum(r['kernel_active_union_us'] for r in per_layer),
                gemm_count=sum(r['gemm_count'] for r in per_layer),kernel_count=sum(r['kernel_count'] for r in per_layer))


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS',reference_primary=str(a.reference),intrusive_diagnostic=str(a.diagnostic),arms={})
    for arm in ('R-NEAR','G-NEAR'):
        root=a.diagnostic/arm
        traces=[json.loads((root/f'phases_repeat-1_rank{r}.json').read_text()) for r in range(4)]
        assert all(len(t)==3072 for t in traces)
        ends=np.array([[max(t[i][stage][1] for t in traces) for stage in ('metadata_plan_complete','forward_global_complete','h2d_global_complete','compute_global_complete','return_global_complete')] for i in range(3072)],np.int64)
        start=np.array([max(t[i]['moe_runtime'][0] for t in traces) for i in range(3072)],np.int64)
        assert np.all(np.diff(ends,axis=1)>=0)
        assert np.all(start[48:]>=ends[47:-1,-1]) and np.all(ends[:,0]>=start)
        pieces=dict(metadata_plan_frontier=(ends[48:,0]-start[48:]).sum()/1e9/63,
                    attention_router_other_frontier=(start[48:]-ends[47:-1,-1]).sum()/1e9/63)
        for col,name in enumerate(('forward_and_rendezvous','h2d_and_rendezvous','compute_and_rendezvous','return_and_rendezvous')):
            pieces[name]=np.diff(ends[48:],axis=1)[:,col].sum()/1e9/63
        calls={}
        for name in ('metadata_call','controller_call','layout_cpu_call','index_pack_call','executor_call'):
            # Skip the one active profiler decode step for the steady CPU-call
            # summary, retaining it separately in the raw complete traces.
            values=np.array([[(t[i][name][1]-t[i][name][0])/1e6 for i in range(48,3072) if not 96<=i<144] for t in traces])
            calls[name]=dict(rank_mean_ms=values.mean(axis=1).tolist(),rank_median_ms=np.median(values,axis=1).tolist(),
                             event_max_mean_ms=float(values.max(axis=0).mean()),note='Nested CPU call spans, may contain device/communication waits; cannot be added to phase frontiers')
        prim=json.loads((a.reference/arm/'result.json').read_text())
        for rank in range(4):
            actual=json.loads((root/f'repeat-1_rank{rank}.json').read_text())
            expected=json.loads((a.reference/arm/f'repeat3_rank{rank}.json').read_text())
            assert actual['tokens']==expected['tokens'] and actual['validation']['state_hash']==expected['validation']['state_hash']
        report['arms'][arm]=dict(primary_statistics=prim['statistics'],diagnostic_phase_frontier_seconds_per_token=pieces,
                                 steady_cpu_calls=calls,reference_token_cache_parity=True,
                                 profiled_kernel_ranks=[kernel_report(root/f'kernel_trace_rank{r}.json') for r in range(4)])
    write(a.out/'BREAKDOWN.json',report)
    lines=['# Stage-I execution breakdown','',
           'Primary performance comes only from the three unprofiled repeats in the reference cohort. Granular runs and one-forward Torch CPU/CUDA traces are separate intrusive diagnostics. All four ranks retain the reference tokens and final cache state exactly.','',
           '| Arm | Primary median TPOT (s) | H2D + rendezvous (s/token, granular diagnostic) | Compute + rendezvous (s/token, granular diagnostic) |','|---|---:|---:|---:|']
    for arm,row in report['arms'].items():
        phases=row['diagnostic_phase_frontier_seconds_per_token']
        lines.append(f"| {arm} | {row['primary_statistics']['TPOT']['median']:.6f} | {phases['h2d_and_rendezvous']:.6f} | {phases['compute_and_rendezvous']:.6f} |")
    lines += ['', '## Metadata and PLAN','',
              'The earlier ~0.25 s/token residual includes attention/router and must not be called metadata overhead. BREAKDOWN.json now separates the global MoE-entry frontier from metadata/PLAN completion. This is a frontier partition, not a sum of CUDA kernels. CPU calls are nested diagnostic spans and must not be summed with phase times. The active profiler step is excluded from the steady CPU-call summary.','',
              '| Arm | Metadata call max-rank mean (ms/layer) | C++ controller call | CPU layout | Index pack |','|---|---:|---:|---:|---:|']
    for arm,row in report['arms'].items():
        c=row['steady_cpu_calls'];lines.append('| '+arm+' | '+' | '.join(f"{c[k]['event_max_mean_ms']:.4f}" for k in ('metadata_call','controller_call','layout_cpu_call','index_pack_call'))+' |')
    lines += ['', '## Expert compute','',
              'The native C++ executor calls separate gate/up/down GEMMs for every expert. Moving the loop to C++ removed per-expert Python crossings but does not remove ATen/cuBLAS launches, per-expert gather/weight operations or small-M inefficiency. Kernel reports correlate CUDA launches inside each executor span to their GPU kernels and report the union of actual kernel-active intervals. CUDA stream windows include host submission gaps; they are not GPU-active time. Profiler CPU overhead prevents a production-speed claim from this one early decode step.','',
              'A separate paired grouped/native probe uses identical packets, cache weights and routing weights, with all demand H2D already complete. Only the original native output advances the model, preserving its entire routing/cache trajectory. Its timing is compute-only, not serving TPOT.','',
              'Ready-First is a different schedule: compute resident/completed experts while other mandatory H2D remains in flight. The serialized mode neutralizes that opportunity because all ranks finish H2D before compute. A separate Ready-First mode must still finish all metadata/forward communication before any H2D and prevent return communication until all H2D and compute are complete. Grouping ready waves trades earlier compute against more launch batches.','',
              'Potential overhead work: native index/layout preparation; reuse metadata/header storage; replace pickle-based checksum preparation with a deterministic native digest while retaining per-event PLAN validation. Removing checks or phase barriers would change the experiment and is not included in these primary results.']
    (a.out/'BREAKDOWN.md').write_text('\n'.join(lines)+'\n')
    write(a.out/'SOURCES.json',{str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for arm in ('R-NEAR','G-NEAR') for p in (a.diagnostic/arm).glob('*.json')})
    print(json.dumps(dict(status='PASS',out=str(a.out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
