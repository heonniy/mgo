#!/usr/bin/env python3
"""Validate and publish the bounded oracle study; missing data never becomes zero."""
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess

from run_rank_oracle_study import ROOT, OUT, PACKAGE, write


def read(path): return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def save(name, rows):
    write(OUT/(name+'.json'),rows)
    if rows:
        with (OUT/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    repeats=[]; ranks=[]; planning=[]; validations=[]; sources=[]; safety=[]
    for status_path in sorted(ROOT.glob('*/status.json')):
        state=read(status_path)
        for sample in state['memory']:
            safety.append(dict(run=state['label'],unix=sample['unix'],host_available_gib=sample['host_available_bytes']/2**30,
                               tree_rss_gib=sample.get('group_rss_bytes',0)/2**30,
                               min_gpu_free_mib=min(g['free_mib'] for g in sample['gpu'].values()),
                               max_gpu_used_mib=max(g['used_mib'] for g in sample['gpu'].values())))
    save('memory_observations',safety)
    for batch in (4,8,16):
        planname=f'plan_b{batch}'; planroot=ROOT/planname/'receipts'
        p=read(planroot/f'{planname}-evidence-rank0.json')
        assert len(p['events'])==65*48
        a=p['assignments']
        assert len(a)==65*48 and all(x['status']=='optimal' for x in a)
        planning.append(dict(batch=batch,events=len(a),solver_seconds=sum(x['solver_seconds'] for x in a),
            prefill_solver_seconds=sum(x['solver_seconds'] for x in a[:48]),
            decode_solver_seconds=sum(x['solver_seconds'] for x in a[48:]),
            max_event_solver_seconds=max(x['solver_seconds'] for x in a),solves=sum(x['solves'] for x in a),
            all_events_optimal=True,trace_sha256=sha(planroot/f'{planname}-evidence-rank0.json')))
        for repeat,order in enumerate(itertools.permutations(('P0','P1','O0'))):
            root=ROOT/f'timing_b{batch}'/'receipts'
            assert read(ROOT/f'timing_b{batch}'/'status.json')['status']=='PASS'
            for position,policy in enumerate(order):
                name=f'b{batch}_{policy}_rep{repeat}'
                rs=[read(root/f'{name}-rep0-rank{r}.json') for r in range(4)]
                es=[read(root/f'{name}-evidence-rank{r}.json') for r in range(4)]
                assert all(x['events']==es[0]['events'] for x in es)
                assert all(x['status']=='PASS' and len(x['rank_step_seconds'])==65 for x in rs)
                assert all(x['metrics']==rs[0]['metrics'] for x in rs)
                for r in range(4):
                    assert rs[r]['generated_token_ids']==es[r]['tokens']
                    baseline=read(root/f'b{batch}_{policy}_rep0-evidence-rank{r}.json')
                    assert es[r]['events']==baseline['events'] and es[r]['tokens']==baseline['tokens']
                    if policy=='O0':
                        plan=read(planroot/f'{planname}-evidence-rank{r}.json')
                        assert es[r]['events']==plan['events'] and es[r]['tokens']==plan['tokens']
                    sources.append(root/f'{name}-rep0-rank{r}.json'); sources.append(root/f'{name}-evidence-rank{r}.json')
                m=rs[0]['metrics']; pre=rs[0]['prefill']['metrics']
                events=es[0]['events']; dec=events[48:]
                survival=read(root/f'{name}-survival.json')
                row=dict(batch=batch,global_batch=4*batch,policy=policy,repeat=repeat,order='-'.join(order),position=position,
                    generation_seconds=rs[0]['global_max_generation_seconds'],tpot_seconds=rs[0]['tpot_seconds'],
                    ttft_seconds=rs[0]['ttft_seconds'],tokens_per_second=rs[0]['fixed_step_output_tokens_per_second'],
                    h2d_bytes=sum(r['host_fetch_bytes'] for r in rs),fetches=m['fetches'],reloads=m['reloads'],
                    remote_pairs=m['remote_token_rank_pairs'],remote_pair_fraction=m['remote_pair_fraction'],
                    decode_remote_pairs=m['remote_token_rank_pairs']-pre['remote_token_rank_pairs'],
                    controller_seconds_max_rank=max(r['controller_seconds'] for r in rs),
                    mean_rank_token_cv=m['mean_event_rank_token_cv'],
                    decode_sum_max_rank_expert_rows=sum(max(e['loads']) for e in dec),
                    decode_peak_max_rank_expert_rows=max(max(e['loads']) for e in dec),
                    decode_sum_all_expert_rows=sum(sum(e['loads']) for e in dec),
                    next_use_survival_fraction=survival['survival_fraction'],next_use_censored=survival['censored'])
                # Payload model: dispatched BF16 hidden rows plus returned BF16
                # expert partials. Metadata is excluded; this is not wire bytes.
                row['peer_activation_payload_bytes']=2048*2*(m['remote_token_rank_pairs']+m['remote_expert_routes'])
                assert row['h2d_bytes']==row['fetches']*9437184
                repeats.append(row)
                validations.append(dict(batch=batch,policy=policy,repeat=repeat,ranks=4,events=3120,
                                        deterministic_repeats=True,oracle_plan_parity=policy=='O0',status='PASS'))
                if repeat==0:
                    for e in events:
                        ranks.append(dict(batch=batch,policy=policy,event=e['event'],step=e['event']//48,layer=e['layer'],
                                          rank0_rows=e['loads'][0],rank1_rows=e['loads'][1],rank2_rows=e['loads'][2],rank3_rows=e['loads'][3],
                                          max_rank_rows=max(e['loads']),total_rows=sum(e['loads']),
                                          distinct_experts=sum(e['distinct'])))
    save('oracle_planning_summary',planning); save('e2e_repeats',repeats); save('rank_load_events',ranks)
    summary=[]
    for batch in (4,8,16):
        for policy in ('P0','P1','O0'):
            rows=[r for r in repeats if r['batch']==batch and r['policy']==policy]
            assert len(rows)==6
            row=dict(batch=batch,policy=policy,repeats=6)
            for key in ('generation_seconds','tpot_seconds','ttft_seconds','tokens_per_second','controller_seconds_max_rank'):
                values=[r[key] for r in rows]
                for label,fun in [('median',statistics.median),('min',min),('max',max)]: row[key+'_'+label]=fun(values)
            for key in ('h2d_bytes','fetches','reloads','remote_pairs','remote_pair_fraction','decode_remote_pairs',
                        'peer_activation_payload_bytes','mean_rank_token_cv','decode_sum_max_rank_expert_rows',
                        'decode_peak_max_rank_expert_rows','decode_sum_all_expert_rows','next_use_survival_fraction','next_use_censored'):
                assert all(r[key]==rows[0][key] for r in rows); row[key]=rows[0][key]
            summary.append(row)
    save('e2e_summary',summary)
    save('rank_load_summary',[{k:r[k] for k in ('batch','policy','mean_rank_token_cv','decode_sum_max_rank_expert_rows','decode_peak_max_rank_expert_rows','decode_sum_all_expert_rows')} for r in summary])
    profiles=read(OUT/'gpu_critical_path.json') if (OUT/'gpu_critical_path.json').exists() else None
    validation=dict(status='PASS' if profiles else 'TIMING_COMPLETE_PROFILES_PENDING',uninstrumented_generations=len(repeats),
                    full_profiles=3 if profiles else 0,planning_cells=3,decode_forwards=64,physical_gpus=[0,1,4,5],
                    checks=validations,smoke=read(OUT/'smoke_validation.json'),solver=read(OUT/'oracle_solver_validation.json'))
    write(OUT/'validation.json',validation)
    report=['# Rank-demand oracle and GPU critical-path study','',
            f'Status: {validation["status"]}. Plan `dccc93c`; physical GPUs 0,1,4,5 only. Exactly 54 unprofiled full generations, three local batches, six policy-order permutations per batch. One prefill plus 64 decode forwards; cache/history reset for every generation. Solver planning time is excluded from frozen-oracle timing.','',
            '## Primary measurements','',
            '| Local B | Policy | E2E median [min, max], s | TPOT median [min, max], s | Decode sum busiest-rank rows | H2D GiB | Remote pairs |',
            '|---:|---|---:|---:|---:|---:|---:|']
    for r in summary:
        report.append(f'| {r["batch"]} | {r["policy"]} | {r["generation_seconds_median"]:.3f} [{r["generation_seconds_min"]:.3f}, {r["generation_seconds_max"]:.3f}] | {r["tpot_seconds_median"]:.4f} [{r["tpot_seconds_min"]:.4f}, {r["tpot_seconds_max"]:.4f}] | {r["decode_sum_max_rank_expert_rows"]:,} | {r["h2d_bytes"]/2**30:.2f} | {r["remote_pairs"]:,} |')
    report+=['','P0 = Balanced Random; P1 = Hungarian-current; O0 = exact rank-demand oracle replay. O0 is an upper-bound diagnostic, not an online deployable speedup. Min/max are observed ranges, not confidence intervals.','', '## Oracle headroom and controls','']
    for batch in (4,8,16):
        r={x['policy']:x for x in summary if x['batch']==batch}; o=r['O0']
        for base in ('P0','P1'):
            b=r[base]
            report.append(f'- B{batch}, O0 relative to {base}: TPOT reduction {100*(1-o["tpot_seconds_median"]/b["tpot_seconds_median"]):+.2f}%; E2E reduction {100*(1-o["generation_seconds_median"]/b["generation_seconds_median"]):+.2f}%; decode busiest-rank row reduction {100*(1-o["decode_sum_max_rank_expert_rows"]/b["decode_sum_max_rank_expert_rows"]):+.2f}%; remote-pair change {100*(o["remote_pairs"]/b["remote_pairs"]-1):+.2f}%; H2D change {100*(o["h2d_bytes"]/b["h2d_bytes"]-1):+.2f}%.')
    report+=['','Policies can change later substitution, routes and fetches. These complete model trajectories do not hold raw demand constant across policies. H2D, reloads, next-use survival and controller time are reported alongside rank load; material differences confound a pure compute-balance interpretation. Pair counts and modeled activation payload exclude NCCL protocol overhead and are not wire traffic measurements.','', '## Solver and validation','']
    for r in planning: report.append(f'- B{r["batch"]}: {r["solver_seconds"]:.3f} s exact solve time across {r["events"]} events; slowest event {r["max_event_solver_seconds"]:.3f} s. Every solve is optimal; no heuristic fallback.')
    report+=['','All six O0 replays reproduce every event route/substitution/admission/victim/cache-slot hash and every full generated token on all ranks. Each policy repeats deterministically. Instrumentation on/off passes the short GPU gate; full B8 profile parity is separately required. CPU exhaustive tests verify both the optimal load and unique lexicographic tie-break.','', '## GPU critical path','']
    if profiles:
        report+=['Decode-only sums of per-event maxima across four ranks; these maxima need not occur on the same rank. Kernel union excludes idle gaps; span includes gaps and waits.','',
                 '| Policy | Expert kernel union max sum (ms) | GEMM union max sum (ms) | Expert span max sum (ms) | NCCL union max sum (ms) | H2D union max sum (ms) |',
                 '|---|---:|---:|---:|---:|---:|']
        totals={}
        for p in ('P0','P1','O0'):
            ps=[x for x in profiles if x['policy']==p and x['step']>0]
            totals[p]={k:sum(x[k] for x in ps) for k in ('max_rank_expert_gpu_union_ms','max_rank_gemm_gpu_union_ms','max_rank_expert_gpu_span_ms','max_rank_nccl_gpu_union_ms','max_rank_h2d_gpu_union_ms')}
            report.append('| '+p+' | '+' | '.join(f'{x:.3f}' for x in totals[p].values())+' |')
        report+=['','The profile includes native worker-thread kernels inside each enclosing MoE layer. Expert ranges include execution preparation and concatenation kernels; the GEMM-only series isolates matrix kernels. H2D copy intervals come from CUPTI, not the asynchronous CPU submission duration. GPU interval attribution must pass expected GEMM counts and exact H2D byte reconciliation. The three profiles follow all primary timing; profile wall times are excluded from speed claims.']
        b8={r['policy']:r for r in summary if r['batch']==8}
        def delta(a,b): return 100*(a/b-1)
        expert='max_rank_expert_gpu_union_ms'
        report+=['','### Answers to the planned questions','',
            f'1. P1 versus P0: decode busiest-rank rows change {delta(b8["P1"]["decode_sum_max_rank_expert_rows"],b8["P0"]["decode_sum_max_rank_expert_rows"]):+.2f}%; measured expert-kernel max-rank time changes {delta(totals["P1"][expert],totals["P0"][expert]):+.2f}%. This is a descriptive physical comparison, with policy-dependent raw routes.',
            f'2. O0 versus P1: expert-kernel max-rank time changes {delta(totals["O0"][expert],totals["P1"][expert]):+.2f}%; versus P0, {delta(totals["O0"][expert],totals["P0"][expert]):+.2f}%. GEMM-only and elapsed-span alternatives are retained above.',
            f'3. At B8, O0 versus P1 changes primary median TPOT {delta(b8["O0"]["tpot_seconds_median"],b8["P1"]["tpot_seconds_median"]):+.2f}% and E2E {delta(b8["O0"]["generation_seconds_median"],b8["P1"]["generation_seconds_median"]):+.2f}%. These use the six unprofiled repetitions, not profile wall time.',
            f'4. O0 versus P1 changes B8 remote pairs {delta(b8["O0"]["remote_pairs"],b8["P1"]["remote_pairs"]):+.2f}% and profiled max-rank NCCL kernel union {delta(totals["O0"]["max_rank_nccl_gpu_union_ms"],totals["P1"]["max_rank_nccl_gpu_union_ms"]):+.2f}%. NCCL intervals include device-side waiting.',
            f'5. O0 versus P1 changes B8 physical H2D volume {delta(b8["O0"]["h2d_bytes"],b8["P1"]["h2d_bytes"]):+.2f}% and median maximum-rank controller time {delta(b8["O0"]["controller_seconds_max_rank_median"],b8["P1"]["controller_seconds_max_rank_median"]):+.2f}%. These concurrent changes and shared-host noise limit attributing all E2E change to compute balance.']
        correlations=[]
        import numpy as np
        from scipy.stats import spearmanr
        for policy in ('P0','P1','O0'):
            ps=[x for x in profiles if x['policy']==policy and x['step']>0]
            x=np.array([v['max_rank_expert_rows'] for v in ps]); y=np.array([v[expert] for v in ps])
            correlations.append(dict(policy=policy,decode_events=len(ps),pearson=float(np.corrcoef(x,y)[0,1]),
                                     spearman=float(spearmanr(x,y).statistic),scope='descriptive across decode layers/steps; not causal'))
        save('row_gpu_correlations',correlations)
        write(OUT/'gpu_phase_summary.json',totals)
    else: report+=['Pending; no GPU critical-path conclusion is made from the row proxy alone.']
    report+=['','## Resource and measurement limits','',
        'All new jobs were sequential R4 jobs. Other users continued work on GPUs 2,3,6,7 and shared CPU, RAM and storage. Balanced ordering reduces fixed order bias but does not remove shared-host interference. No long-horizon quality conclusion is made from this fixed-work numeric workload.',
        'Timing retains compact CPU evidence and planned-row counting for all policies, plus frozen-demand checks for O0. Hashing, serialization and token checks run outside the generation timer. No NVTX/CUDA events/CUPTI are active in primary timing. See IMPLEMENTATION.md for exact boundaries and memory guards.',
        f'Observed minimum host availability: {min(x["host_available_gib"] for x in safety):.1f} GiB; minimum selected-GPU free memory: {min(x["min_gpu_free_mib"] for x in safety):,} MiB. Initial smoke RSS observations included only the launcher process group; full planning/timing monitoring counts descendants as well.',
        '', '## Reproduction and artifacts','',
        'Launcher: `scripts/run_rank_oracle_study.py` stages smoke, planning, timing, profiles. Analysis: `scripts/analyze_rank_oracle_profiles.py`, `scripts/summarize_rank_oracle_study.py`. Raw traces remain at `/home/hwlee/mgo-results/rank_demand_oracle_20261002`; giant Nsight traces are not committed.',
        'Inspect `e2e_repeats.csv`, `e2e_summary.csv`, `rank_load_summary.csv`, `rank_load_events.csv`, `oracle_planning_summary.csv`, `gpu_phase_profile.csv`, `gpu_critical_path.csv`, `validation.json` and hash receipts. CSVs supply figure data.',
        '', 'No joint communication/load policy, production controller optimization, substitution retuning, replication or migration was added.']
    (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    sources+=list(ROOT.glob('plan_b*/receipts/*evidence*.json'))+list(ROOT.glob('profile_b*/receipts/*.json'))+list(ROOT.glob('*/status.json'))
    write(OUT/'raw_receipts.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(sources))])
    write(OUT/'artifact_hashes.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps({k:v for k,v in validation.items() if k not in ('checks','smoke','solver')}))

if __name__=='__main__': main()
