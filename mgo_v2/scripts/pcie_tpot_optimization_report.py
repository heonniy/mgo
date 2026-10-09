"""Validate common R/G optimization cohort and partition diagnostic TPOT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from pcie_host import write
from pcie_policy_report import summarize_arm,expected_quota
from validate_pcie_phases import validate

VARIANTS=('NATIVE','GROUPED','GROUPED_META')
BASES=('R-NEAR','G-NEAR')
STAGES=('metadata_plan_complete','forward_global_complete','h2d_global_complete',
        'compute_global_complete','return_global_complete')
CALLS=('plan_event_call','metadata_call','controller_call','layout_cpu_call','index_pack_call','executor_call')

def partition(root):
    traces=[json.loads((root/f'phases_repeat-1_rank{r}.json').read_text()) for r in range(4)]
    ranks=[json.loads((root/f'repeat-1_rank{r}.json').read_text()) for r in range(4)]
    assert all(len(t)==3072 for t in traces)
    ends=np.array([[max(t[i][s][1] for t in traces) for s in STAGES] for i in range(3072)],np.int64)
    entry=np.array([max(t[i]['moe_runtime'][0] for t in traces) for i in range(3072)],np.int64)
    assert np.all(np.diff(ends,axis=1)>=0) and np.all(entry[48:]>=ends[47:-1,-1])
    assert np.all(ends[:,0]>=entry)
    names=('metadata_plan','forward','h2d','compute','return','attention_router_other')
    values=np.column_stack((ends[48:,0]-entry[48:],np.diff(ends[48:],axis=1),entry[48:]-ends[47:-1,-1]))/1e9
    first=max(r['first_ns'] for r in ranks);last=max(r['end_ns'] for r in ranks)
    boundary=(last-ends[-1,-1])-(first-ends[47,-1])
    phases=dict(zip(names,values.sum(axis=0)/63))
    phases['attention_router_other']+=boundary/1e9/63
    tpot=(last-first)/1e9/63
    assert abs(sum(phases.values())-tpot)<1e-9 and phases['attention_router_other']>=0
    cpu=np.array([[[((t[i][c][1]-t[i][c][0])/1e6) for c in CALLS] for i in range(48,3072)] for t in traces])
    residual=cpu[:,:,0]-cpu[:,:,1:5].sum(axis=2)
    assert residual.min()>-1e-6,'Nested PLAN spans unexpectedly overlap'
    stats={c:dict(rank_mean_ms=cpu[:,:,col].mean(axis=1).tolist(),
                  max_rank_mean_ms=float(cpu[:,:,col].max(axis=0).mean())) for col,c in enumerate(CALLS)}
    stats['plan_residual_checksum_slots_glue']=dict(rank_mean_ms=residual.mean(axis=1).tolist(),
                  max_rank_mean_ms=float(residual.max(axis=0).mean()),
                  note='PLAN total minus metadata, controller, layout, index-pack; includes checksum, binding, glue and wrapper overhead')
    with (root/'diagnostic_frontiers.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['event',*names])
        for event,row in enumerate(values,48):writer.writerow([event,*row])
    return dict(TPOT=tpot,seconds_per_token=phases,nested_cpu_calls=stats,
                attention_boundary_adjustment_seconds_per_token=boundary/1e9/63,
                scope='Separate timers-only diagnostic, no Torch profiler/record_function; endpoint partition includes rendezvous')

def plot(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    x=np.arange(3);width=.34;colors={'R-NEAR':'#0072B2','G-NEAR':'#D55E00'}
    for j,base in enumerate(BASES):
        summaries=[report['arms'][base+'__'+v]['live']['primary']['TPOT'] for v in VARIANTS]
        med=np.array([s['median'] for s in summaries])
        axes[0].bar(x+(j-.5)*width,med,width,label=base,color=colors[base])
        axes[0].errorbar(x+(j-.5)*width,med,yerr=np.array([[s['median']-s['min'] for s in summaries],
                     [s['max']-s['median'] for s in summaries]]),fmt='none',ecolor='#333333',capsize=3)
    axes[0].set_xticks(x,['Native','Grouped','Grouped +\nC++ metadata'])
    axes[0].set_ylabel('Unprofiled median TPOT (s/token)');axes[0].legend()
    labels=[base+'\n'+v.replace('GROUPED_META','Grouped+meta').replace('GROUPED','Grouped').replace('NATIVE','Native')
            for v in VARIANTS for base in BASES]
    keys=('metadata_plan','forward','h2d','compute','return','attention_router_other')
    bottom=np.zeros(6)
    for key,color in zip(keys,('#CC79A7','#56B4E9','#E69F00','#009E73','#0072B2','#777777')):
        y=np.array([report['arms'][b+'__'+v]['diagnostic']['seconds_per_token'][key] for v in VARIANTS for b in BASES])
        axes[1].bar(np.arange(6),y,bottom=bottom,color=color,label=key.replace('_',' '));bottom+=y
    axes[1].set_xticks(np.arange(6),labels,rotation=35,ha='right',fontsize=8)
    axes[1].set_ylabel('Separate diagnostic TPOT (s/token)')
    axes[1].legend(fontsize=7,ncol=2,loc='upper right')
    for ax in axes:ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('R-NEAR / G-NEAR: common decode optimizations, full64 generation')
    fig.savefig(out/'tpot_optimization.pdf');fig.savefig(out/'tpot_optimization.png',dpi=160);plt.close(fig)

def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    cohort=json.loads((root/'result.json').read_text());assert cohort['status']=='PASS' and cohort['sequence']=='optimize'
    arms=[b+'__'+v for v in VARIANTS for b in BASES];assert cohort['arms']==arms
    report=dict(status='PASS',root=str(root),arms={},placement_comparisons={},implementation_comparisons={})
    for arm in arms:
        path=root/arm;validate(path,(-1,));live=summarize_arm(path)
        for rank in range(4):
            for repeat in range(1,cohort['primary_repeats']+1):
                r=json.loads((path/f'repeat{repeat}_rank{rank}.json').read_text())
                assert r['triton_no_compile'] and r['no_compile']
            r=json.loads((path/f'repeat1_rank{rank}.json').read_text())
            diagnostic=json.loads((path/f'repeat-1_rank{rank}.json').read_text())
            assert r['tokens']==diagnostic['tokens'] and r['validation']['state_hash']==diagnostic['validation']['state_hash']
            assert np.array_equal(np.load(path/f'policy_trace_repeat1_rank{rank}.npy'),np.load(path/f'policy_trace_repeat-1_rank{rank}.npy'))
            if arm.endswith('GROUPED_META'):
                warm=json.loads((path/f'metadata_repeat0_rank{rank}.json').read_text());assert warm['wire_checks']==48 and warm['calls']==3024
            if not arm.endswith('NATIVE'):
                warm=json.loads((path/f'grouped_repeat0_rank{rank}.json').read_text())
                assert len(warm['checks'])==48 and all(x['finite'] and x['relative_l2']<=.01 for x in warm['checks'])
        trace=np.load(path/'policy_trace_repeat1_rank0.npy')[48:]
        count=trace[:,19].astype(np.int64)
        r=np.array([expected_quota(int(m),int(event),False) for m,event in zip(count,trace[:,52])])
        g=np.array([expected_quota(int(m),int(event),True) for m,event in zip(count,trace[:,52])])
        critical_r=r.reshape(-1,2,2).sum(axis=2).max(axis=1)
        critical_g=g.reshape(-1,2,2).sum(axis=2).max(axis=1)
        count_proxy=dict(events_with_different_critical_group_counts_fraction=float(np.mean(critical_r!=critical_g)),
                         mean_critical_group_count_reduction_percent=float(np.mean((critical_r-critical_g)/np.maximum(1,critical_r)))*100,
                         scope='Count-only same-M counterfactual on this arm\'s live trace; not measured H2D speedup')
        report['arms'][arm]=dict(live=live,diagnostic=partition(path),same_m_quota_count_proxy=count_proxy)
    for v in VARIANTS:
        r=report['arms']['R-NEAR__'+v]['live']['primary']['TPOT']['median']
        g=report['arms']['G-NEAR__'+v]['live']['primary']['TPOT']['median']
        report['placement_comparisons'][v]=dict(R_median=r,G_median=g,G_reduction_percent=(r-g)/r*100)
    for base in BASES:
        a=[report['arms'][base+'__'+v]['live']['primary']['TPOT']['median'] for v in VARIANTS]
        report['implementation_comparisons'][base]=dict(native_to_grouped_reduction_percent=(a[0]-a[1])/a[0]*100,
                                                       grouped_to_metadata_reduction_percent=(a[1]-a[2])/a[1]*100)
        agreement=[];prefix_lengths=[]
        for rank in range(4):
            rows=[json.loads((root/(base+'__'+v)/f'repeat1_rank{rank}.json').read_text()) for v in VARIANTS]
            assert np.array_equal(np.array(rows[0]['tokens'])[:,0],np.array(rows[1]['tokens'])[:,0]),'Native prefill first token changed'
            assert rows[1]['tokens']==rows[2]['tokens'] and rows[1]['validation']['state_hash']==rows[2]['validation']['state_hash']
            same=np.array(rows[0]['tokens'])==np.array(rows[1]['tokens'])
            agreement.append(float(same.mean()))
            prefix_lengths.extend(np.where(same.all(axis=1),64,same.argmin(axis=1)).tolist())
            assert json.loads((root/f'metadata_parity_rank{rank}.json').read_text())['status']=='PASS'
        report['implementation_comparisons'][base]['native_grouped_token_agreement_rank']=agreement
        report['implementation_comparisons'][base]['native_grouped_matching_prefix_lengths']=prefix_lengths
        report['implementation_comparisons'][base]['native_grouped_first_token_exact']=True
        report['implementation_comparisons'][base]['grouped_metadata_exact_token_cache_trace_parity']=True
    report['notes']=['Only unprofiled repeats supply primary TPOT. Separate diagnostic is a frontier partition, not kernel-active time.',
       'Each arm generates its own greedy tokens. Native versus grouped BF16 reductions may change routes and H2D; report agreement and live bytes.',
       'C++ metadata changes preserve exact grouped tokens, policy trace and cache state.',
       'The common cohort retains the same 8.25-MiB grouped activation workspace per rank in every arm; expert-cache slots remain unchanged.',
       'R/G both receive each optimization. A positive placement benefit is not assumed. Short cohorts do not establish a universal gain.',
       'Full 54 GiB per-NUMA shared pinned source, C30/P2, no prefetch, no expert H2D/token A2A overlap or H2D/compute overlap.']
    write(out/'RESULTS.json',report)
    lines=['# TPOT optimization: G-NEAR versus R-NEAR','',*report['notes'],'',
           '| Common implementation | R median TPOT (s) | G median TPOT (s) | G reduction (%) |','|---|---:|---:|---:|']
    for v,c in report['placement_comparisons'].items():lines.append(f"| {v} | {c['R_median']:.6f} | {c['G_median']:.6f} | {c['G_reduction_percent']:.3f} |")
    lines+=['','| Arm | Metadata/PLAN | Forward | H2D | Compute | Return | Attention/router/other |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in report['arms'].items():lines.append('| '+arm+' | '+' | '.join(f'{v:.6f}' for v in row['diagnostic']['seconds_per_token'].values())+' |')
    lines+=['','All phase values above are seconds/token from the separate timers-only diagnostic. PLAN residual includes checksum exchange, slot binding and glue. Nested CPU calls in RESULTS.json cannot be added to the frontier phase totals.','',
            '| Arm | Metadata max-rank mean ms/layer | PLAN residual ms/layer |','|---|---:|---:|']
    for arm,row in report['arms'].items():
        c=row['diagnostic']['nested_cpu_calls'];lines.append(f"| {arm} | {c['metadata_call']['max_rank_mean_ms']:.4f} | {c['plan_residual_checksum_slots_glue']['max_rank_mean_ms']:.4f} |")
    lines+=['','## Numerical trajectories and fetch opportunity','',
            'All variants retain native prefill and identical first generated tokens for each placement. Grouped decode can diverge later through different BF16 GEMM reductions; token agreement is reproducibility evidence, not task accuracy. Grouped and grouped+metadata retain the entire same trajectory exactly.','',
            '| Placement | Native/grouped token agreement (global %) | Grouped/metadata tokens, cache, trace |','|---|---:|---|']
    for base,row in report['implementation_comparisons'].items():
        lines.append(f"| {base} | {np.mean(row['native_grouped_token_agreement_rank'])*100:.3f} | Exact |")
    lines+=['','| Arm | Decode H2D (GiB) | Mean M | M mod 4 = 2 (%) | Count-only critical-group reduction (%) |','|---|---:|---:|---:|---:|']
    for arm,row in report['arms'].items():
        d=row['live']['live_decode'][0];c=row['same_m_quota_count_proxy']
        lines.append(f"| {arm} | {d['H2D_bytes']/2**30:.3f} | {d['miss_mean']:.3f} | {d['M_mod_4_equals_2_fraction']*100:.3f} | {c['mean_critical_group_count_reduction_percent']:.3f} |")
    lines+=['','The last column holds each event\'s M fixed and compares quota counts only. For M=54 the critical group changes 28 to 27 fetches (3.57% fewer); for the microbenchmark M=6 it changes 4 to 3 (25% fewer). This count comparison cannot replace measured H2D timing. Live cache trajectories and total bytes are reported separately above.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    plot(report,out)
    write(out/'SOURCES.json',{str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.npy','.csv')})
    print(json.dumps(dict(status='PASS',out=str(out))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
