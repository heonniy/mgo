#!/usr/bin/env python3
"""Build a bounded mechanistic report from validated scientific receipts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from summarize_trajectory_study import OUT,ROOT,PACKAGE,read,POLICIES,save


def main():
    validation=read(OUT/'validation.json');assert validation['status']=='PASS'
    trajectory=read(OUT/'trajectory_summary.json');replays=read(OUT/'matched_replay_summary.json')
    survival=read(OUT/'next_use_survival.json');aligned=read(ROOT/'cumulative_event_deltas.json')
    comparisons=[];operations=[];decomposition=[]
    for batch in (4,8,16):
        complete=read(ROOT/'replay'/f'b{batch}'/'complete.json')['summaries']
        for source in POLICIES:
            tr={p:next(r for r in trajectory if r['mode']=='replay' and r['batch']==batch and r['source_policy']==source and r['policy']==p) for p in POLICIES}
            ti={p:next(r for r in replays if r['batch']==batch and r['source_policy']==source and r['policy']==p) for p in POLICIES}
            sv={p:next(r for r in survival if r['mode']=='replay' and r['batch']==batch and r['source_policy']==source and r['policy']==p) for p in POLICIES}
            row=dict(batch=batch,source_policy=source)
            for key in ('remote_pairs','miss','reloads','evictions','candidate_count','changed_coverage_layers','rank_token_cv','rank_token_max_mean'):
                row['delta_'+key]=tr['hungarian_current'][key]-tr['random'][key]
                row['percent_'+key]=100*(tr['hungarian_current'][key]/max(tr['random'][key],1e-30)-1)
            row['delta_h2d_gib']=row['delta_miss']*9437184/2**30
            row['delta_controller_seconds']=ti['hungarian_current']['controller_seconds_median']-ti['random']['controller_seconds_median']
            row['delta_survival_pp']=100*(sv['hungarian_current']['next_use_survival']-sv['random']['next_use_survival'])
            group=[r for r in aligned if r['batch']==batch and r['source_policy']==source]
            row['first_fetch_divergence_event']=next((r['event'] for r in group if r['delta_miss']!=0),None)
            row['first_remote_divergence_event']=next((r['event'] for r in group if r['delta_remote_pairs']!=0),None)
            row['cumulative_fetch_min']=min(r['cumulative_miss'] for r in group)
            row['cumulative_fetch_max']=max(r['cumulative_miss'] for r in group)
            row['events_with_fetch_divergence']=sum(r['delta_miss']!=0 for r in group)
            row['sum_absolute_event_fetch_delta']=sum(abs(r['delta_miss']) for r in group)
            row['max_absolute_cumulative_fetch_delta']=max(abs(r['cumulative_miss']) for r in group)
            comparisons.append(row)
            for policy in POLICIES:
                measured=[r for r in complete if r['source_policy']==source and r['policy']==policy]
                base=measured[0]
                admission_keys=('admission_prep','admission_cost_build','admission_assignment','admission_result','admission_policy')
                operations.append(dict(batch=batch,source_policy=source,policy=policy,
                    victim_selections=base['counts']['choose_calls'],
                    candidate_entries=base['counts']['eviction_candidate_count'],
                    mean_candidates_per_victim=base['counts']['eviction_candidate_count']/base['counts']['choose_calls'],
                    coverage_sync_calls=base['counts']['choose_calls'],
                    coverage_added=base['counts']['coverage_resident_additions'],coverage_removed=base['counts']['coverage_resident_removals'],
                    coverage_changed_layers=base['counts']['coverage_changed_layers'],
                    cache_key_list_calls=base['counts']['keys_on_rank_calls'],cache_keys_materialized=base['counts']['cache_keys_materialized'],
                    rank_cost_elements=base['counts']['hungarian_rank_cost_elements'],assignment_elements=base['counts']['hungarian_assignment_elements'],
                    admission_fraction_percent=statistics.median(100*sum(r['times_ns'].get(k,0) for k in admission_keys)/r['controller_ns'] for r in measured),
                    cost_and_solver_fraction_percent=statistics.median(100*sum(r['times_ns'].get(k,0) for k in ('admission_cost_build','admission_assignment'))/r['controller_ns'] for r in measured),
                    coverage_ranking_fraction_percent=statistics.median(100*r['times_ns'].get('coverage_rank_and_victim',0)/r['controller_ns'] for r in measured),
                    diagnostic_accounting_fraction_percent=statistics.median(100*r['times_ns'].get('diagnostic_accounting',0)/r['controller_ns'] for r in measured)))
    for batch in (4,8,16):
        cells={(r['source_policy'],r['policy']):r for r in trajectory if r['mode']=='replay' and r['batch']==batch}
        times={(r['source_policy'],r['policy']):r for r in replays if r['batch']==batch}
        A=('random','random');B=('random','hungarian_current');C=('hungarian_current','random');D=('hungarian_current','hungarian_current')
        for metric in ('miss','reloads','remote_pairs','candidate_count','controller_seconds_median'):
            data=times if metric=='controller_seconds_median' else cells
            a,b,c,d=[data[k][metric] for k in (A,B,C,D)]
            decomposition.append(dict(batch=batch,metric=metric,random_on_random=a,current_on_random=b,random_on_current=c,current_on_current=d,
                own_trajectory_delta=d-a,policy_delta_random_demand=b-a,raw_stream_delta_current_policy=d-b,
                raw_stream_delta_random_policy=c-a,policy_delta_current_demand=d-c))
    save('counterfactual_decomposition',decomposition)
    save('matched_comparisons',comparisons);save('operation_counts',operations)
    audit=['# Controller implementation audit','',
        'This packet measures existing work; it does not optimize or redesign the controller. Instrumented copies preserve every original policy statement, verified by an AST identity check and stateful CPU/GPU parity gates. Normal controller/admission/eviction/runtime files and the native extension are unchanged.','',
        'Every physical rank gathers the same global metadata and independently computes the entire global plan. All 74,880 rank-event plan/cache hashes match across ranks. Four times the logical work is executed across CPU processes, but summing rank times is not an E2E critical-path estimate.','',
        'Coverage calls `keys_on_rank()` for every victim, materializing the rank cache list before removing pinned keys. Candidate gate scores and damage ranks are rebuilt in Python for every chosen victim. `_sync_coverage()` constructs global resident sets and updates affected layers. Admission uses Python token-source/base-destination loops and allocates an incoming×world rank-cost matrix, then an incoming×incoming quota-expanded matrix for SciPy. Substitution repeatedly constructs expert/source groups and searches legal anchors. The unchanged cache consistency check scans all residents each event.','',
        'Allocation and loop work is included in the surrounding spans; there is no allocator profiler. The table measures list materializations/candidate visits, not Python memory allocation counts. Diagnostic set-difference counter work is separately timed and must not be mistaken for original controller cost. No GPU synchronization occurs inside these timers.','',
        '| B | Raw source | Policy | Victims | Candidate visits | Candidates/victim | Cache keys materialized | Admission share | Cost+solver share | Coverage ranking share | Diagnostic accounting share |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in operations:
        audit.append(f"| {r['batch']} | {r['source_policy']} | {r['policy']} | {r['victim_selections']:,} | {r['candidate_entries']:,} | {r['mean_candidates_per_victim']:.1f} | {r['cache_keys_materialized']:,} | {r['admission_fraction_percent']:.2f}% | {r['cost_and_solver_fraction_percent']:.2f}% | {r['coverage_ranking_fraction_percent']:.2f}% | {r['diagnostic_accounting_fraction_percent']:.2f}% |")
    audit+=['','Shares are medians of five per-repetition fractions of instrumented controller wall time. They are descriptive CPU diagnostics, not fractions of uninstrumented GPU generation time. [All operation counts](operation_counts.csv) include global-resident changes, changed coverage layers and matrix sizes.','',
        'Potential later optimization targets are candidate scoring/ranking, repeated resident/list/set construction and replicated planning. They require a separate owner-reviewed stage; none is applied here.']
    (OUT/'implementation_audit.md').write_text('\n'.join(audit)+'\n')
    h2max=max(r['cost_and_solver_fraction_percent'] for r in operations)
    largest_fetch_percent=max(abs(r['percent_miss']) for r in comparisons)
    h1directions=dict(fewer=sum(r['delta_miss']<0 for r in comparisons),more=sum(r['delta_miss']>0 for r in comparisons),equal=sum(r['delta_miss']==0 for r in comparisons))
    lines=['# Four-GPU admission trajectory and controller breakdown','',
        'The requested bounded diagnostic packet is complete: six R4 physical cells on GPUs **0,1,4,5**, followed by 60 single-process CPU replays. No R8 job was launched. All instrumentation/output/cache/fetch/replay correctness gates pass. The existing five-repeat uninstrumented study remains the performance evidence.','',
        f"Under matched raw demand, Current changes the future cache trajectory: {h1directions['fewer']} of six source-stream/batch comparisons have fewer total fetches and {h1directions['more']} have more. Hungarian cost construction plus assignment occupies at most **{h2max:.2f}%** of instrumented controller time. The largest absolute matched-demand net fetch change is **{largest_fetch_percent:.3f}%**. Per-comparison magnitudes and timing variability below matter more than a general win/loss label.",'',
        '## Protocol and evidence boundaries','',
        'Plan commit `30614bbc065c140060ef31af9bc4f8615ab16ca5`; prior results `e61758e`; measurement implementation `caf0143`. Qwen3-30B-A3B-Instruct-2507 BF16; cache30, gate .20/similarity .65, Coverage W128/k1/lambda2, hard expert-count quotas, seed42, no replication/migration. B4/B8/B16 are local batches (global 16/32/64). Each full run executes one prefill plus 64 decode forwards, producing 65 fixed-work tokens even after EOS.','',
        'A separate opt-in module wraps unchanged policy statements with exclusive CPU spans. The normal execution path has no diagnostic hooks. Smoke checks compare on/off full generated tokens, slots, cache and metrics for both policies on all four ranks. Full R4/B8 generated tokens also match every corresponding prior uninstrumented repeat. This does not constitute a new long-horizon quality evaluation.','',
        'Each batch uses both its Random and Current raw global router streams. On each stream, both policies start with empty independent cache/history/RNG and run five times: 3 batches × 2 streams × 2 policies × 5 repetitions = 60. Own-policy replay is a subset of this matrix and reproduces every physical plan/cache hash. Raw demand stays fixed; effective routes may change through substitution. These counterfactuals do not regenerate hidden states or test model quality.','',
        'Three independent single-process replay workers use distinct CPU affinities 180/182/184; each performs its own matrix serially with one numeric-library thread. Some CPU replay overlaps later GPU diagnostics. Other user jobs occupy GPUs 2,3,6,7 and share host resources. Repetition ranges reflect observed timing variability, not confidence intervals. No new diagnostic wall time is used as a speedup claim. Repetition zero additionally captures hashes/events outside the timed spans; later repetitions write timing/count records and verify the same final state and operation totals.','',
        'W128 covers global token rows, so decode histories span 8/4/2 events per layer at local B4/B8/B16. Next-use distances count global layer events; adjacent demand at the same layer is usually 48 events apart. Survival denominator includes only admissions with a later raw demand; all end-of-stream censoring is reported. Owner changes mean re-admission on another rank after eviction, never resident migration.','',
        'Physical H2D bytes are per-event native dispatcher fetch counts × 9 MiB/expert, validated against miss operations and run totals. No new CUPTI profile is collected. The unchanged executor and prior physical transfer audits support the interpretation; CPU replay bytes are predicted logical fetch volume.','',
        '## Existing uninstrumented evidence','',
        'The prior Random/Current medians show different directions across cells. R8 entries are retrospective context only. The new B4/B16 R4 probes do not have five-repeat uninstrumented timing in this packet.','',
        '| World / local B | Random E2E median [min, max] (s) | Current E2E median [min, max] (s) | Current change |','|---|---:|---:|---:|']
    prior=read(OUT/'retrospective_summary.json')
    for world,batch in sorted({(r['world'],r['local_batch']) for r in prior}):
        group={r['policy']:r for r in prior if r['world']==world and r['local_batch']==batch}
        a,b=[group[p]['generation_seconds'] for p in POLICIES]
        lines.append(f"| R{world} / B{batch} | {a:.3f} [{group['random']['generation_seconds_min']:.3f}, {group['random']['generation_seconds_max']:.3f}] | {b:.3f} [{group['hungarian_current']['generation_seconds_min']:.3f}, {group['hungarian_current']['generation_seconds_max']:.3f}] | {100*(b/a-1):+.2f}% |")
    lines+=['','Positive changes mean slower. The R4/B8 ranges overlap and Current varies substantially across five repeats; the +4.72% median is not proof of a stable causal slowdown. [Retrospective table](retrospective_summary.csv) includes TPOT, controller time, physical H2D, reloads, remote pairs and rank load. [Descriptive correlations](retrospective_correlations.json) are confounded by world/batch and repeated observations; no causal interpretation.','',
        '## Physical trajectories','',
        'One diagnostic generation per condition. These rows can differ in generated routing demand between policies; they are not the matched-demand causal comparison.','',
        '| Local B | Policy | Fetches | Evictions | Reloads | H2D (GiB) | Remote pairs | Mean token CV | Max/mean token load |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in trajectory:
        if r['mode']=='physical':
            lines.append(f"| {r['batch']} | {r['policy']} | {r['miss']:,} | {r['evictions']:,} | {r['reloads']:,} | {r['physical_h2d_bytes']/2**30:.2f} | {r['remote_pairs']:,} | {r['rank_token_cv']:.3f} | {r['rank_token_max_mean']:.3f} |")
    lines+=['','[Trajectory summary](trajectory_summary.csv) also reports exact/substitution hits, candidate visits, changed coverage layers, decode-step eviction/H2D rates and effective expert GEMM row imbalance. Complete per-event/rank trajectories and raw router metadata remain on the server.','',
        '## Matched raw demand: Current minus Random','',
        'Negative count/byte deltas mean less work for Current. Controller deltas subtract five-repeat total-time medians. Counts are deterministic; the cumulative timing plot instead sums event-wise five-repeat medians, which need not equal the median of totals.','',
        '| Local B | Raw source | Δ remote pairs | Δ fetches | Δ reloads | Δ H2D (GiB) | Δ controller (s) | Δ survival (pp) | First fetch divergence |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['batch']} | {r['source_policy']} | {r['delta_remote_pairs']:+,} | {r['delta_miss']:+,} | {r['delta_reloads']:+,} | {r['delta_h2d_gib']:+.2f} | {r['delta_controller_seconds']:+.3f} | {r['delta_survival_pp']:+.2f} | {r['first_fetch_divergence_event']} |")
    lines+=['','### Fixed-demand policy effect versus raw-stream effect','',
        'The 2×2 replay isolates two descriptive steps: Random→Current policy on the Random raw stream, then Random→Current raw stream under Current policy. Their sum equals the own-trajectory difference. The reverse path is also retained in the data. Raw-stream changes contain model/substitution feedback; these are systems counterfactuals, not hidden-state interventions.','',
        '| B | Metric | Own Current − own Random | Policy effect on Random demand | Stream effect under Current |',
        '|---:|---|---:|---:|---:|']
    for r in decomposition:
        if r['metric'] in ('miss','controller_seconds_median'):
            lines.append(f"| {r['batch']} | {r['metric']} | {r['own_trajectory_delta']:+.3f} | {r['policy_delta_random_demand']:+.3f} | {r['raw_stream_delta_current_policy']:+.3f} |")
    lines+=['','Full two-direction accounting: [counterfactual decomposition](counterfactual_decomposition.csv).']
    lines+=['','![Cumulative event-aligned differences](cumulative_trajectories.png)','',
        'Initial owners and communication differ before future fetch counts need diverge. “First fetch divergence” is the first global event with unequal admissions, including prefill. [Matched comparisons](matched_comparisons.csv) retain minimum/maximum cumulative fetch difference; [event deltas](cumulative_event_deltas.csv) expose changes of direction rather than hiding them in endpoint totals.','',
        '## Controller work and repeated CPU times','',
        '| Local B | Raw source | Policy | Controller median [min, max] (s) | Cost build median (s) | Assignment median (s) | Coverage ranking median (s) |',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in replays:
        lines.append(f"| {r['batch']} | {r['source_policy']} | {r['policy']} | {r['controller_seconds_median']:.3f} [{r['controller_seconds_min']:.3f}, {r['controller_seconds_max']:.3f}] | {r.get('admission_cost_build_seconds_median',0):.3f} | {r.get('admission_assignment_seconds_median',0):.3f} | {r.get('coverage_rank_and_victim_seconds_median',0):.3f} |")
    lines+=['','![Controller component times](controller_components.png)','',
        'Exclusive spans are not GPU time. Timer/context-manager overhead remains in totals; explicit coverage-counter set operations have their own diagnostic accounting category. Route serialization, trajectory calculation and hashing lie outside `controller_ns`, but inside physical diagnostic wall time. Thus the diagnostic total is not an unbiased estimate of the prior uninstrumented controller. [Implementation audit](implementation_audit.md) separates measured operation counts from proposed follow-up opportunities. No controller optimization was applied.','',
        '## Next-use survival and concrete state examples','',
        '![Admission next-use survival](next_use_survival.png)','',
        '[Next-use summary](next_use_survival.csv) reports admissions, later demands, censored admissions, survival, exact/substitution/reload outcomes and owner changes on re-admission. Every admission-level record is retained server-side. [State examples](state_examples.md) contain at least five favorable and five unfavorable fetch-divergence events, with full cache snapshots, gate scores, independently computed Coverage damage and subsequent exact/substitute service until eviction. They explain concrete trajectories; they are not a new admission objective.','',
        '## Communication savings and rank load','',
        '![Matched-demand communication and load changes](communication_load.png)','',
        'Each point aggregates one decode step across 48 layer events. The load numerator sums each layer’s maximum rank token count; it is a workload proxy, not measured GPU compute duration. The same raw token demand can use fewer remote pairs while assigning more token work to the busiest rank. Effective expert-route/GEMM-row counts are retained separately.','',
        '## Hypotheses and limits','',
        '**H1 — future trajectory matters:** policy-dependent cache trajectories are verified under matched raw demand, but material net downstream savings must be assessed per cell rather than inferred from any nonzero difference. Endpoint cancellation, small count differences and timing variability can leave the historical E2E explanation unresolved. The decomposition below distinguishes a placement-policy effect on fixed demand from the different raw streams produced by physical generation. These counterfactuals do not imply identical model hidden states.','',
        f'**H2 — the assignment solver is not the main controller cost:** supported within these diagnostics. Cost build plus solve is at most {h2max:.2f}% of total controller time. Candidate ranking/victim selection and other downstream work dominate. Timing noise and instrumentation prevent assigning the entire historical E2E difference to one CPU component.','',
        '**H3 — expert-count quotas do not ensure token-work balance:** the event data measure that discrepancy directly, but a causal explanation of the historical R4/B8 slowdown from skew alone remains **inconclusive**. No new isolated GPU execution profile was collected; aggregate CV or correlation cannot separate compute skew, host/controller waiting and overlap. Do not add a load constraint based solely on this packet.','',
        '**H4 — direct NVSwitch byte saving is secondary:** remains limited to prior resident-only evidence. The earlier R4 fixed-event map sweeps had small/non-monotonic latency responses and varying rank compute loads. This packet does not independently isolate communication time and does not turn byte savings into additive E2E percentages.','',
        '## Validation and reproduction','',
        '26 CPU tests pass, including 320 stateful diagnostic on/off events. Eight rank/policy smoke receipts verify identical generated tokens/cache/metrics. All six full GPU cells contain 3,120 layer events; 24 rank receipts contain 74,880 rank events. Per-event physical fetches match admissions, every rank agrees on plans/cache, and own-policy CPU replay reproduces all physical event hashes. Forty full-output comparisons bind new B8 diagnostics to the prior five-repeat baseline. Five fresh-state repetitions in all 12 replay conditions preserve final cache/history/plan and operation totals.','',
        'Source/input/checkpoint/device binding: [manifest](measurement_manifest.json), [provenance](provenance.json), [validation](validation.json), [artifact hashes](artifact_hashes.json). Raw data root: `/home/hwlee/mgo-results/admission_trajectory_controller_breakdown_20261002`. Launch and analysis scripts are in `mgo_v2/scripts`; the measurement source hashes are frozen in the manifest. Raw trusted pickle router streams must only be loaded from this controlled directory.','',
        'Stop condition reached. Admission, Coverage, load constraints, same+path, replication/migration and controller optimization remain unchanged pending owner review.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
