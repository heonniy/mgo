#!/usr/bin/env python3
"""Validate and summarize the bounded diagnostic and matched-demand packet."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path('/home/hwlee/mgo-results/admission_trajectory_controller_breakdown_20261002')
OUT=PACKAGE/'experiments/admission_trajectory_controller_breakdown_20261002'
EXPERT_BYTES=9437184
POLICIES=('random','hungarian_current')

def read(path): return json.loads(path.read_text())
def events(path): return [json.loads(line) for line in path.open()]
def save(name, rows):
    json_root=ROOT if name=='cumulative_event_deltas' else OUT
    (json_root/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        with (OUT/f'{name}.csv').open('w') as f:
            columns=list(dict.fromkeys(k for row in rows for k in row))
            w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader()
            w.writerows({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)

def next_use(rows):
    pending={};resident={};last_owner={};output=[];owner_changes=0
    for row in rows:
        i,layer=row['event'],row['layer']
        substitutes=dict(row['substitutes'])
        for expert in row['raw_sources']:
            key=(layer,expert)
            if key in pending:
                admission=pending.pop(key)
                alive=key in resident
                output.append(dict(admission_event=admission[0],layer=layer,expert=expert,admitted_rank=admission[1],
                    next_demand_event=i,distance=i-admission[0],survived=alive,
                    outcome='exact_hit' if alive else ('substitution_covered' if expert in substitutes else 'reload'),
                    owner_changed=last_owner.get(key)!=admission[1],censored=False))
        for layer,expert,rank,slot in row['evictions']:
            assert resident.pop((layer,expert))==rank
        for layer,expert,rank,slot in row['admissions']:
            key=(layer,expert)
            owner_changes+=key in last_owner and last_owner[key]!=rank
            resident[key]=rank;last_owner[key]=rank
            assert key not in pending
            pending[key]=(i,rank)
    for (layer,expert),(i,rank) in pending.items():
        output.append(dict(admission_event=i,layer=layer,expert=expert,admitted_rank=rank,
            next_demand_event=None,distance=None,survived=None,outcome='right_censored',owner_changed=None,censored=True))
    assert len(output)==sum(len(r['admissions']) for r in rows)
    eligible=[r for r in output if not r['censored']]
    return output,dict(admissions=len(output),later_demand=len(eligible),right_censored=len(output)-len(eligible),
        next_use_survival=sum(r['survived'] for r in eligible)/max(1,len(eligible)),
        next_exact_hit=sum(r['outcome']=='exact_hit' for r in eligible),
        next_substitution_covered=sum(r['outcome']=='substitution_covered' for r in eligible),
        next_reload=sum(r['outcome']=='reload' for r in eligible),
        median_next_demand_distance=statistics.median(r['distance'] for r in eligible) if eligible else None,
        owner_changes_on_readmission=owner_changes,
        next_use_owner_changed=sum(r['owner_changed'] for r in eligible))

def summarize_rows(rows):
    for r in rows:
        incoming=np.bincount([a[2] for a in r['admissions']],minlength=4).tolist()
        assert incoming==r['quotas'] and max(incoming)-min(incoming)<=1
        assert sum(incoming)==r['miss']==len(r['admissions'])
        assert r['counts'].get('choose_calls',0)==len(r['evictions'])
        assert r['calls'].get('coverage_sync',0)==len(r['evictions'])
        assert sum(r['rank_tokens'])==r['remote_pairs']+r['local_pairs']
        assert sum(r['rank_expert_rows'])>=sum(r['rank_tokens'])
        assert r['remote_expert_routes']>=r['remote_pairs']
        assert all(free>=0 for free in r['free_after'])
    decode=[r for r in rows if r['phase']=='decode']
    totals={k:sum(r[k] for r in rows) for k in ('hit','subhit','miss','reloads','remote_pairs','local_pairs','remote_expert_routes')}
    totals.update(events=len(rows),admissions=sum(len(r['admissions']) for r in rows),evictions=sum(len(r['evictions']) for r in rows),
        logical_fetch_bytes=totals['miss']*EXPERT_BYTES,
        rank_token_cv=statistics.mean(r['rank_token_cv'] for r in rows),
        rank_token_max_mean=statistics.mean(r['rank_token_max_mean'] for r in rows),
        mean_rank_expert_row_max_mean=statistics.mean(max(r['rank_expert_rows'])/max(1,statistics.mean(r['rank_expert_rows'])) for r in rows),
        candidate_count=sum(r['counts'].get('eviction_candidate_count',0) for r in rows),
        changed_coverage_layers=sum(r['counts'].get('coverage_changed_layers',0) for r in rows),
        decode_sum_max_rank_tokens=sum(max(r['rank_tokens']) for r in decode),
        decode_sum_max_rank_expert_rows=sum(max(r['rank_expert_rows']) for r in decode),
        decode_fetches_per_step=sum(r['miss'] for r in decode)/64,
        decode_h2d_bytes_per_step=sum(r['miss'] for r in decode)*EXPERT_BYTES/64,
        decode_evictions_per_step=sum(len(r['evictions']) for r in decode)/64,
        reload_per_admission=totals['reloads']/max(1,totals['miss']))
    return totals


def main():
    assert read(ROOT/'status.json')['status']=='PHYSICAL_COMPLETE'
    breakdown=[];trajectory=[];survival=[];replay_summary=[];validation=[];aligned=[];all_repeats=[];physical_receipts=[]
    for batch in (4,8,16):
        for policy in POLICIES:
            stem=f'b{batch}_{policy}'
            rows=events(ROOT/'physical'/f'{stem}-events-rank0.jsonl')
            assert len(rows)==3120
            ranks=[];actual_bytes=0
            for rank in range(4):
                receipt=read(ROOT/'physical'/f'{stem}-rep0-rank{rank}.json')
                rr=events(ROOT/'physical'/f'{stem}-events-rank{rank}.jsonl')
                pp=events(ROOT/'physical'/f'{stem}-physical-rank{rank}.jsonl')
                assert len(rr)==len(pp)==3120
                assert all(r['event']==i and r['layer']==i%48 for i,r in enumerate(rr))
                assert all(a['plan_sha256']==b['plan_sha256'] and a['cache_sha256']==b['cache_sha256'] for a,b in zip(rows,rr))
                assert all(p['fetches']==sum(a[2]==rank for a in r['admissions']) for p,r in zip(pp,rows))
                assert sum(p['physical_h2d_bytes'] for p in pp)==receipt['host_fetch_bytes']
                actual_bytes+=receipt['host_fetch_bytes']
                assert receipt['status']=='PASS' and receipt['world']==4 and receipt['local_batch']==batch
                assert len(receipt['generated_token_ids'])==batch and all(len(t)==65 for t in receipt['generated_token_ids'])
                ranks.append(receipt)
                physical_receipts.append(receipt)
                labels=sorted({k for r in rr for k in r['times_ns']})
                for label in labels:
                    breakdown.append(dict(mode='physical',batch=batch,source_policy=policy,policy=policy,rank=rank,
                        component=label,seconds=sum(r['times_ns'].get(label,0) for r in rr)/1e9,
                        calls=sum(r['calls'].get(label,0) for r in rr),repeats=1))
                for r in rr:
                    assert r['controller_ns']==sum(r['times_ns'].values())+r['timer_unattributed_ns']
                    assert r['timer_unattributed_ns']>=0
                validation.append(dict(check='physical_rank_event_plan_cache_fetch_and_timer',batch=batch,policy=policy,rank=rank,status='PASS'))
            totals=summarize_rows(rows)
            assert actual_bytes==totals['logical_fetch_bytes']
            for key,expected in [('miss',totals['miss']),('fetches',totals['miss']),('reloads',totals['reloads']),('remote_token_rank_pairs',totals['remote_pairs'])]:
                assert all(r['metrics'][key]==expected for r in ranks)
            trajectory.append(dict(mode='physical',batch=batch,source_policy=policy,policy=policy,physical_h2d_bytes=actual_bytes,**totals))
            detail,stats=next_use(rows)
            (ROOT/'physical'/f'{stem}-next-use.json').write_text(json.dumps(detail,separators=(',',':')))
            survival.append(dict(mode='physical',batch=batch,source_policy=policy,policy=policy,**stats))
        complete=read(ROOT/'replay'/f'b{batch}'/'complete.json')
        assert complete['status']=='PASS' and complete['replays']==20
        all_repeats.extend(complete['summaries'])
        for source in POLICIES:
            paired={};timings={}
            for policy in POLICIES:
                stem=f'{source}-to-{policy}'
                rows=events(ROOT/'replay'/f'b{batch}'/f'{stem}-rep0-events.jsonl')
                assert len(rows)==3120
                paired[policy]=rows
                group=[r for r in complete['summaries'] if r['source_policy']==source and r['policy']==policy]
                assert len(group)==5
                common_counts=set.intersection(*(set(r['counts']) for r in group))
                assert all(len({r['counts'][key] for r in group})==1 for key in common_counts), 'Replay operation counts changed'
                assert all(r['final']==group[0]['final'] for r in group), 'Replay final states changed'
                labels=sorted({k for r in group for k in r['times_ns']})
                total_med=statistics.median(r['controller_ns'] for r in group)/1e9
                result=dict(batch=batch,source_policy=source,policy=policy,repeats=5,own_trajectory=source==policy,
                    controller_seconds_median=total_med,
                    controller_seconds_min=min(r['controller_ns'] for r in group)/1e9,
                    controller_seconds_max=max(r['controller_ns'] for r in group)/1e9,
                    admissions=group[0]['final']['admissions'],evictions=group[0]['final']['evictions'])
                for label in labels:
                    values=[r['times_ns'].get(label,0)/1e9 for r in group]
                    result[label+'_seconds_median']=statistics.median(values)
                    breakdown.append(dict(mode='replay',batch=batch,source_policy=source,policy=policy,rank='single_process',
                        component=label,seconds=statistics.median(values),minimum=min(values),maximum=max(values),repeats=5))
                replay_summary.append(result)
                trajectory.append(dict(mode='replay',batch=batch,source_policy=source,policy=policy,**summarize_rows(rows)))
                detail,stats=next_use(rows)
                (ROOT/'replay'/f'b{batch}'/f'{stem}-next-use.json').write_text(json.dumps(detail,separators=(',',':')))
                survival.append(dict(mode='replay',batch=batch,source_policy=source,policy=policy,**stats))
                if source==policy:
                    physical=events(ROOT/'physical'/f'b{batch}_{source}-events-rank0.jsonl')
                    assert all(a['plan_sha256']==b['plan_sha256'] and a['cache_sha256']==b['cache_sha256'] for a,b in zip(rows,physical))
                raw=[events(ROOT/'replay'/f'b{batch}'/f'{stem}-rep{rep}-timings.jsonl') for rep in range(5)]
                assert all(len(r)==3120 for r in raw)
                timings[policy]=[dict(controller=statistics.median(r[i]['controller_ns'] for r in raw)/1e9,
                    coverage=statistics.median(r[i]['times_ns'].get('coverage_sync',0) for r in raw)/1e9,
                    eviction=statistics.median(sum(v for k,v in r[i]['times_ns'].items() if k in ('eviction_candidates','coverage_sync','coverage_sync_dispatch','coverage_rank_and_victim','eviction_choose','cache_keys_on_rank')) for r in raw)/1e9) for i in range(3120)]
                validation.append(dict(check='replay_five_repeats_and_own_trace_parity',batch=batch,source_policy=source,policy=policy,status='PASS'))
            cumulative=defaultdict(float)
            for i,(a,b) in enumerate(zip(paired['random'],paired['hungarian_current'])):
                delta={key:b[key]-a[key] for key in ('remote_pairs','reloads','miss')}
                delta['evictions']=len(b['evictions'])-len(a['evictions'])
                delta['h2d_bytes']=delta['miss']*EXPERT_BYTES
                for key in ('controller','coverage','eviction'):
                    delta[key+'_seconds']=timings['hungarian_current'][i][key]-timings['random'][i][key]
                for key,value in delta.items(): cumulative[key]+=value
                aligned.append(dict(batch=batch,source_policy=source,event=i,step=i//48,layer=i%48,phase='prefill' if i<48 else 'decode',
                    **{'delta_'+k:v for k,v in delta.items()},**{'cumulative_'+k:v for k,v in cumulative.items()},
                    delta_rank_token_cv=b['rank_token_cv']-a['rank_token_cv'],
                    delta_max_rank_tokens=max(b['rank_tokens'])-max(a['rank_tokens'])))
    boots=[read(ROOT/'physical'/f'provenance-rank{r}.json') for r in range(4)]
    assert [p['boot']['visible_gpu'] for p in boots]==['0','1','4','5']
    assert all(p['boot']['strict_numa'] and p['boot']['numa_policy']=='membind-strict' for p in boots)
    assert len({p['code_sha256'] for p in boots})==1
    state=read(ROOT/'status.json')
    for relative,expected in state['source_sha256'].items():
        assert hashlib.sha256((PACKAGE/relative).read_bytes()).hexdigest()==expected, ('measurement source drift',relative)
    for name,expected in state['input_sha256'].items():
        assert hashlib.sha256((Path('/home/hwlee/mgo-results/runtime_validation_20261001')/name).read_bytes()).hexdigest()==expected
    expected_uuids=['GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d','GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb',
                    'GPU-6076e2f2-5b63-3761-5586-56ceb7df8139','GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730']
    observed=read(ROOT/'physical_device_observations.json')['rows']
    assert len(observed)==4 and all(r['world']==4 and r['gpu_uuid']==expected_uuids[r['rank']] for r in observed)
    previous_boot=read(Path('/home/hwlee/mgo-results/local_remote_e2e_impact_20261001/stage_b_r4/provenance-rank0.json'))
    assert boots[0]['checkpoint']==previous_boot['checkpoint']
    assert all(p['input_sha256'][name]==previous_boot['input_sha256'][name] for p in boots for name in ('similarity','affinity','workload'))
    for policy in POLICIES:
        for rank in range(4): assert read(ROOT/'physical'/f'b4_{policy}-smoke-rank{rank}.json')['status']=='PASS'
    # B8 generated-token identity against all prior five-repeat uninstrumented runs.
    prior=Path('/home/hwlee/mgo-results/local_remote_e2e_impact_20261001/stage_b_r4')
    for policy,label in [('random','C0_balanced_random'),('hungarian_current','C1_hungarian_current')]:
        for rank in range(4):
            actual=read(ROOT/'physical'/f'b8_{policy}-rep0-rank{rank}.json')['generated_token_ids']
            for repeat in range(5):
                expected=read(prior/f'b8_{label}-rep{repeat}-rank{rank}.json')['generated_token_ids']
                assert actual==expected, ('B8 baseline output mismatch',policy,rank,repeat)
                assert read(ROOT/'physical'/f'b8_{policy}-rep0-rank{rank}.json')['metrics']==read(prior/f'b8_{label}-rep{repeat}-rank{rank}.json')['metrics']
    save('controller_breakdown',breakdown);save('trajectory_summary',trajectory);save('next_use_survival',survival)
    save('matched_replay_summary',replay_summary)
    save('replay_repeats',all_repeats)
    (OUT/'physical_rank_receipts.json').write_text(json.dumps(physical_receipts,indent=2)+'\n')
    save('cumulative_event_deltas',aligned)
    (OUT/'validation.json').write_text(json.dumps(dict(status='PASS',physical_cells=6,rank_receipts=24,
        global_physical_events=18720,rank_events=74880,cpu_replays=60,
        replay_events=187200,smoke_rank_checks=8,baseline_b8_token_comparisons=40,checks=validation),indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(physical=boots,device_observations=read(ROOT/'physical_device_observations.json')),indent=2)+'\n')
    print('PASS: six physical cells, 60 CPU replays, all event/plan/cache/fetch/timer/token gates.')

if __name__=='__main__': main()
