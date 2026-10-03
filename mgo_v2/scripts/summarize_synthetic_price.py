#!/usr/bin/env python3
"""C2 gate, conditional C3/C4, and bounded final result."""
from synthetic_price_cpu import P,ROOT,OLD,sha,receipt,write,csvfile
from synthetic_price_gate import timing_gate,RHOS
from price_envelope import serial_envelope
from fractions import Fraction
from pathlib import Path
import json,statistics,sys,time
import numpy as np

def main():
    manifest=json.loads((P/'execution_manifest.json').read_text())
    for group in ('source_sha256','input_sha256'):
        for path,h in manifest[group].items():assert sha(path)==h,path
    resource=json.loads((P/'resource_price_sweep.json').read_text());freeze=json.loads((P/'frozen_rho_trace_summary.json').read_text())
    state=json.loads((P/'C2.json').read_text());assert state['status']=='PASS' and len(state['cells'])==10
    for r in state['receipts']:assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    smoke=json.loads((ROOT/'C2/smoke_T0/status.json').read_text());assert smoke['status']=='PASS' and smoke['transports']==['P2P/IPC']
    rows=[];pairs=[];memory=[]
    for cell in state['cells']:
        root=ROOT/'C2'/cell['label'];ranks=[json.loads((root/f'rank{r}.json').read_text()) for r in range(4)]
        status=json.loads((root/'status.json').read_text());assert status['status']=='PASS';memory+=status['memory']
        f=next(f for f in freeze['schedules'] if f['rho']==cell['rho'])
        for rank,r in enumerate(ranks):
            assert r['status']=='PASS' and r['rank']==rank and r['rho']==cell['rho'] and r['pass_index']==cell['pass_index']
            assert r['transport_env']=={'NCCL_CUMEM_ENABLE':'0'} and not r['nccl_info_in_timing']
            assert r['no_model'] and r['no_h2d'] and r['no_cache_controller'] and r['validation_outside_timing'] and r['allocations_outside_timing']
            assert r['condition_order']==(['actual','self'] if cell['pass_index']==0 else ['self','actual'])
            assert r['actual_counts_sha256']==f['actual']['sha256'] and r['self_counts_sha256']==f['self']['sha256']
        pair=dict(pass_index=cell['pass_index'],rho=cell['rho'])
        for kind in ('actual','self'):
            conditions=[r['conditions'][kind] for r in ranks]
            assert all(c['warmup_full_traces']==1 and c['timed_full_traces']==5 and c['events']==384 and c['collective_calls_per_trace']==768 and c['payload_valid'] for c in conditions)
            assert all(len(c['repeats'])==5 for c in conditions)
            peer=sum(c['peer_bytes_per_trace'] for c in conditions)
            assert peer==(f['decode']['peer_activation_bytes'] if kind=='actual' else 0)
            assert all(r['conditions']['actual']['self_bytes_per_trace']==r['conditions']['self']['self_bytes_per_trace'] for r in ranks)
            samples=[]
            for iteration in range(5):
                repeats=[c['repeats'][iteration] for c in conditions];assert all(r['iteration']==iteration and r['payload_valid'] and len(r['events_ms'])==384 for r in repeats)
                samples.append(dict(iteration=iteration,whole_cuda_ms=max(r['whole_cuda_ms'] for r in repeats),whole_wall_ms=max(r['whole_wall_ms'] for r in repeats),sum_event_max_rank_ms=sum(max(r['events_ms'][i]['pair'] for r in repeats) for i in range(384)),max_rank_sum_event_ms=max(sum(e['pair'] for e in r['events_ms']) for r in repeats)))
            medians={k:statistics.median(r[k] for r in samples) for k in samples[0] if k!='iteration'}
            row=dict(pass_index=cell['pass_index'],rho=cell['rho'],kind=kind,peer_bytes=peer,self_bytes=sum(c['self_bytes_per_trace'] for c in conditions),**medians,samples=samples)
            rows.append(row);pair[kind+'_ms']=medians['whole_cuda_ms']
        pair['remote_ms']=pair['actual_ms']-pair['self_ms'];pairs.append(pair)
    assert [(r['pass_index'],r['rho']) for r in state['cells']]==[(p,r) for p in (0,1) for r in (RHOS if p==0 else tuple(reversed(RHOS)))]
    gate=timing_gate(pairs)
    csvfile(P/'comm_actual_self_timing.csv',[{k:v for k,v in r.items() if k!='samples'} for r in rows])
    write(P/'comm_actual_self_timing.json',dict(status='PASS',rows=rows,pairs=pairs,gate=gate,aggregation='median of five whole-trace max-rank CUDA intervals; remote = actual median minus self median, unmodified'))
    crossovers=[dict(model='resource',price_numerator=r['price']['numerator'],price_denominator=r['price']['denominator'],price=r['price']['decimal'],tied_rhos=r['winners']) for r in resource['envelope']['crossovers']]
    time_models={};cost_models=[];sweep=[];model_points=[]
    if gate['passed']:
        calibration=next(r for r in resource['provenance']['source_receipts'] if r['path'].endswith('break_even_microcost.json'))
        assert sha(calibration['path'])==calibration['sha256']
        micro=json.loads(Path(calibration['path']).read_text());h=next(r for r in micro['rows'] if r['kind']=='h2d' and r['case']=='four-rank-concurrent')
        matrix=json.loads((P/'matrix.json').read_text())
        for metric in ('median','p90'):
            cost=h[metric+'_ms'];assert cost==matrix['h2d_existing_ms']['concurrent4_'+metric]
            lines=[]
            for f in freeze['schedules']:
                with Path(f['actual']['path']).open() as src:events=json.load(src)['events']
                counts=[max(e['rank_fetch_counts']) for e in events]
                assert sum(counts)==f['sum_event_max_rank_fetches']
                modeled=Fraction(str(cost))*sum(counts)
                cost_models.append(dict(model=metric,rho=f['rho'],one_copy_ms=cost,sum_event_max_rank_fetches=sum(counts),modeled_H2D_ms=float(modeled)))
                pp=[r for r in pairs if r['rho']==f['rho']]
                own=Fraction(str(statistics.median(r['self_ms'] for r in pp)))
                raw_remote=statistics.median(r['remote_ms'] for r in pp)
                remote=Fraction(0) if f['rho']==.75 else Fraction(str(raw_remote))
                intercept=modeled+own
                lines.append((f['rho'],intercept,remote))
                model_points.append(dict(model=metric,rho=f['rho'],H2D_ms=float(modeled),self_ms=float(own),remote_ms=float(remote),raw_remote_ms=raw_remote,zero_peer_remote_fixed=f['rho']==.75))
            envelope=serial_envelope(lines);time_models[metric]=envelope
            for gamma in matrix['gamma_grid']:
                values=[a+gamma*b for _,a,b in lines];minimum=min(values)
                for (rho,_,_),value in zip(lines,values):sweep.append(dict(model=metric,gamma=gamma,rho=rho,J_time_ms=float(value),winner=value==minimum))
            crossovers += [dict(model='time_'+metric,price_numerator=r['price']['numerator'],price_denominator=r['price']['denominator'],price=r['price']['decimal'],tied_rhos=r['winners']) for r in envelope['crossovers']]
        csvfile(P/'h2d_cost_model.csv',cost_models);write(P/'h2d_cost_model.json',dict(rows=cost_models,calibration=calibration,not_measured_E2E=True))
        csvfile(P/'time_price_sweep.csv',sweep);write(P/'time_price_sweep.json',dict(rows=sweep,coefficients=model_points,envelopes=time_models,synthetic_only=True))
        shifted=any(r['winners']!=[0] for r in time_models['median']['regions'])
        decision='TIME_PRICE_SHIFT' if shifted else 'NO_TIME_PRICE_SHIFT'
    else:
        decision='TIMING_UNSTABLE'
        assert not any((P/name).exists() for name in ('h2d_cost_model.json','time_price_sweep.json'))
    csvfile(P/'policy_crossover.csv',crossovers);write(P/'policy_crossover.json',dict(resource=resource['envelope'],time=time_models or None,C2_gate_passed=gate['passed']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,timing)=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    boundaries=[r['price']['decimal'] for r in resource['envelope']['crossovers']]
    prices=np.geomspace(1,4096,500)
    for point in resource['points']:
        cost=point['expert_h2d_bytes']+prices*point['peer_activation_bytes'];ax.plot(prices,cost/2**30,label=f"rho={point['rho']}")
    for x in boundaries:
        ax.axvline(x,linestyle=':',color='#777777',alpha=.5)
        ax.text(x,.97,f'{x:.2f}',transform=ax.get_xaxis_transform(),ha='right',va='top',rotation=90,fontsize=8,backgroundcolor='white')
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set(xlabel='Synthetic byte price lambda',ylabel='J_byte (GiB equivalents)',title='Exact resource-price shift');ax.legend(fontsize=8);ax.grid(alpha=.2)
    for p,style in ((0,'o-'),(1,'s--')):
        selected=sorted([r for r in pairs if r['pass_index']==p],key=lambda r:r['rho'])
        timing.plot([r['rho'] for r in selected],[r['remote_ms'] for r in selected],style,label=f'pass {p+1}')
    timing.axhline(0,color='#777777',linewidth=.7);timing.set(xlabel='Replica budget rho',ylabel='Actual minus self CUDA median (ms)',title=f'C2: {decision}');timing.legend();timing.grid(alpha=.2)
    fig.savefig(P/'crossover.png',dpi=180);fig.savefig(P/'crossover.svg');plt.close(fig)
    svg=P/'crossover.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    lines=['# Synthetic communication-price crossover','',f'**{decision}**. The exact CPU result remains **RESOURCE_PRICE_SHIFT**.','',
        'Only the original B8/cache30 five-rho frontier was used. C0 source outputs, raw capture sizes/hashes and historical producer source blobs were verified. C1 imported the original hashed greedy replica implementation and reproduced all full/prefill/decode totals and final cache hashes; 2,160 action/state events matched the independent frozen applier.','',
        '## Exact byte resource prices','', '| Transition rho | Exact lambda | Decimal lambda |','|:---|:---|---:|']
    for r in resource['envelope']['crossovers']:
        x=r['price'];lines.append(f"| {' ↔ '.join(map(str,r['winners']))} | {x['numerator']}/{x['denominator']} | {x['decimal']:.6f} |")
    first=resource['envelope']['crossovers'][0]
    lines+=['',f"The first transition is lambda={first['price']['decimal']:.6f}: one peer byte must be priced at that many H2D byte-equivalents for rho={first['winners'][1]} to tie F. The optimal budget then increases through .25, .5 and .75. This dimensionless resource-price result is not a latency or PCIe/NVLink bandwidth ratio.",'',
        '## Actual versus matched self-only timing','', '| Pass | rho | Actual CUDA ms | Self CUDA ms | Remote premium ms |','|---:|---:|---:|---:|---:|']
    for r in sorted(pairs,key=lambda r:(r['pass_index'],r['rho'])):lines.append(f"| {r['pass_index']+1} | {r['rho']} | {r['actual_ms']:.6f} | {r['self_ms']:.6f} | {r['remote_ms']:.6f} |")
    lines+=['','Each median uses five full-trace max-rank CUDA intervals after one warmup. All twenty cells retained exactly 768 collective calls and their original self rows. Self-only zeroed off-diagonal counts. Buffers/events were preallocated and payload checks stayed outside timing. Raw negative residuals are not clipped. Wall and event aggregations are secondary in the timing artifacts.','',
        '## Predeclared timing gate','', '| Pass | Check | rho | Passed |','|---:|:---|---:|:---|']
    for r in gate['checks']:lines.append(f"| {r['pass_index']+1} | {r['check']} | {r['rho']} | {r['passed']} |")
    lines+=['','The owner selected normalization by same-pass rho0 remote premium. Each adjacent nonzero-traffic rho must have normalized premium ≤1.10× its predecessor, with nonnegative premiums and a defined positive rho0 denominator. The zero-remote rho=.75 actual/self control must agree within 10% of self.','',
        '![Crossover and timing gate](crossover.png)','']
    if not gate['passed']:
        zero_checks=[r for r in gate['checks'] if r['check']=='zero_remote_control_agreement']
        lines += [f"The zero-remote controls differ by {zero_checks[0]['relative_difference']:+.2%} and {zero_checks[1]['relative_difference']:+.2%} in the two passes, despite identical traffic counts. Negative premiums also occur for nonzero-traffic policies. Actual-minus-self is therefore not a validated remote-cost estimate in this run.",'']
        lines+=['C2 failed, so **no C3 H2D time model or C4 gamma sweep is accepted or published**. The exact C0 shift is established, but the experiment cannot assign it a validated time-price crossover. No extra repetitions, NCCL tuning, R3 or new H2D measurement were run.']
    else:
        lines+=['C2 passed. C3 reuses the existing 9-MiB concurrent-four median/p90 calibration and multiplies it by each event’s maximum rank fetch count. C4 uses median-of-two-pass self and remote coefficients; zero-peer rho=.75 remote cost is fixed to zero after control agreement, while the raw residual is preserved. These are modeled critical paths, not measured E2E.','', '| Model | Crossover | Tied rho |','|:---|---:|:---|']
        for r in crossovers:
            if r['model'].startswith('time_'):lines.append(f"| {r['model']} | {r['price']:.6f} | {r['tied_rhos']} |")
    peak=max(s.get('group_rss_bytes',0) for s in memory);minimum=min(s['host_available_bytes'] for s in memory)
    lines+=['','## Boundaries and validation','',
        '- R3 remains historical unstable context only. Prior pass/order-dependent R3 trace and microcost results do not calibrate gamma. There is no R3 rerun or physical PCIe equivalence claim.',
        '- One new T0 smoke selected P2P/IPC. Ten paired worker invocations ran twenty timing cells, 100 timed full traces and twenty warmups; all payload checks passed. No model/H2D/cache/controller computation entered timing.',
        f"- C1 CPU peak RSS {freeze['peak_rss_mib']:.2f} MiB under the 4-GiB address-space bound; C2 peak sampled process-tree RSS {peak/2**30:.2f} GiB, minimum host available {minimum/2**30:.2f} GiB. No OOM occurred.",
        '- Six exact-envelope/gate unit tests passed. Source and receipt hashes were checked. Our model workers on GPU 0/1/4/5 were restored; GPU 2/3/6/7 and their jobs were untouched.',
        '- Stop after this bounded result. No extra policy, longer trace, batch/cache sweep, substitution, model capture or physical F/K/C run follows.','',
        'See [resource prices](resource_price_sweep.json), [frozen schedules](frozen_rho_trace_summary.json), [timings and gates](comm_actual_self_timing.json), [crossovers](policy_crossover.json), [validation](validation.json), and [execution conventions](EXECUTION_PROTOCOL.md).','']
    (P/'RESULTS.md').write_text('\n'.join(lines))
    load=Path('/home/hwlee/mgo-results/model_inference_load_20261003');workers=[]
    for w in state['restored_workers']:
        assert w['status']=='STARTED' and w['gpu'] in (0,1,4,5)
        assert 'model_inference_load.py' in Path(f"/proc/{w['pid']}/cmdline").read_bytes().decode()
        r=json.loads((load/f"gpu{w['gpu']}.json").read_text());assert r['pid']==w['pid'] and r['batch']==1024 and time.time()-r['unix']<120
        workers.append({k:r[k] for k in ('gpu','pid','batch','iterations','unix')})
    assert {r['gpu'] for r in workers}=={0,1,4,5} and 'torch' not in sys.modules
    validation=dict(status='PASS',decision=decision,resource_decision=resource['decision'],timing_gate_passed=gate['passed'],timing_gate=gate,CPU_action_state_events=2160,timing_cells=20,timed_traces=100,warmup_traces=20,transport_smokes=1,new_h2d_measurements=0,new_R3_runs=0,new_model_runs=0,unit_tests=6,source_hashes_unchanged=True,C1_peak_rss_mib=freeze['peak_rss_mib'],C2_peak_sampled_group_rss_bytes=peak,C2_min_host_available_bytes=minimum,workers_restored=workers,summary_source_sha256=sha(Path(__file__)),output_sha256={f.name:sha(f) for f in P.iterdir() if f.suffix in ('.json','.csv','.svg','.png') and f.name!='validation.json'}|{'RESULTS.md':sha(P/'RESULTS.md')})
    write(P/'validation.json',validation)
    print(json.dumps({k:validation[k] for k in ('status','decision','resource_decision','timing_gate_passed','timing_cells')},indent=2))
if __name__=='__main__':main()
