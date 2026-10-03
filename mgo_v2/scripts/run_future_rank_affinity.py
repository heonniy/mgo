#!/usr/bin/env python3
"""Bounded sequential CPU placement cells and deterministic verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import argparse,gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from replica_pareto_cpu import IndependentZeroReplay
from future_rank_affinity import PlacementReplay, POLICIES, owner_costs
from run_rank_local_reuse import source, aggregate, write, receipt, CAPS
P=Path(__file__).resolve().parents[1]/'experiments/future_rank_affinity_placement_20261003'
ROOT=Path('/home/hwlee/mgo-results/future_rank_affinity_placement_20261003')

def stable(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()

def run(batch,verify):
    start=time.monotonic();events,demand,provenance=source(batch)
    basepath=P/f'B{batch}.json'
    if verify:
        baseline=json.loads(basepath.read_text());assert baseline['status']=='PASS'
    else:assert not basepath.exists(),'No unauthorized repeats'
    result=dict(status='RUNNING',batch=batch,provenance=provenance,points=[])
    refs={};future={}
    for policy,horizon in POLICIES.items():
        replay=PlacementReplay(CAPS,policy,future,demand)
        independent=IndependentZeroReplay(CAPS) if policy=='F' else None
        rows=[];digest=hashlib.sha256()
        path=ROOT/f'B{batch}_{policy}_{"verify" if verify else "primary"}.jsonl.gz'
        with gzip.open(path,'wt') as out:
            for i,(layer,origins,selected) in enumerate(events):
                row,dest,tr,ops=replay.step(i,layer,origins,selected)
                state=replay.state()
                if independent:
                    st,ds,dispatch,combine,first,reloads=independent.event(layer,origins,selected)
                    assert state==st and np.array_equal(dest,ds)
                    assert np.array_equal(tr['dispatch'],dispatch) and np.array_equal(tr['combine'],combine)
                    assert (row['first_copy_fetches'],row['reload_fetches'])==(first,reloads)
                    if i>=48:refs[i]=dest.copy()
                assert row['replica_fetches']==row['duplicate_slots']==0
                digest.update(stable(dict(event=i,state=state,dest=dest.tolist(),row=row,ops=ops,decisions=replay.event_decisions)))
                if i>=48:
                    out.write(json.dumps(dict(event=i,step=i//48,layer=layer,row=row,decisions=replay.event_decisions,state_sha256=hashlib.sha256(stable(state)).hexdigest()),separators=(',',':'))+'\n')
                rows.append(row)
        totals=aggregate(rows[48:]);decisions=replay.decisions;life=replay.finish()
        eligible=[r for r in life if r['next_demand_event'] is not None]
        changed=[r for r in eligible if r['changed']]
        point=dict(batch=batch,policy=policy,**totals,
            local_service_fraction=sum(r['final_local_services'] for r in rows[48:])/sum(r['raw_expert_routes'] for r in rows[48:]),
            mean_unique_per_rank=np.mean([r['rank_unique'] for r in rows[48:]],axis=0).tolist(),
            peak_unique_per_rank=np.max([r['rank_unique'] for r in rows[48:]],axis=0).tolist(),
            evictions_per_rank=replay.evictions,owner_choice_histogram=[sum(d['owner']==r for d in decisions) for r in range(4)],
            owner_decisions=len(decisions),owner_changes=sum(d['changed'] for d in decisions),
            fraction_owner_differs_F=sum(d['changed'] for d in decisions)/len(decisions),
            next_demand_eligible=len(eligible),survived_next_demand=sum(r['survived_next_demand'] for r in eligible),
            fraction_survived_next_demand=sum(r['survived_next_demand'] for r in eligible)/len(eligible) if eligible else None,
            changed_next_demand_eligible=len(changed),changed_survived_next_demand=sum(r['survived_next_demand'] for r in changed),
            admissions_without_observed_next_demand=len(life)-len(eligible),
            owner_attributable_peer_bytes_saved=sum(r['owner_attributable_peer_bytes_saved'] for r in rows[48:]),
            deterministic_sha256=digest.hexdigest(),
            steps=[dict(step=s,policy=policy,**{k:sum(r[k] for r in rows[s*48:(s+1)*48]) for k in ('expert_h2d_bytes','peer_activation_bytes','reload_fetches','owner_changes')}) for s in range(1,9)])
        assert len(decisions)==totals['total_fetches']==len(life)
        if policy=='F':
            old=json.loads((P.parent/'rank_local_reuse_oracle_20261003'/f'R02_B{batch}.json').read_text())['baseline']
            assert all(totals[k]==v for k,v in old.items()),(totals,old)
            if batch==8:assert (totals['total_fetches'],totals['expert_h2d_bytes'],totals['peer_activation_bytes'],totals['remote_token_rank_pairs'])==(17635,166424739840,334970880,23179)
            # Frozen F destination effect costs for future-only same-layer steps.
            effects={}
            for i in range(48,432):
                _,o,s=events[i]
                for e in np.unique(s):effects[i,int(e)]=owner_costs(o,s,refs[i],int(e))
            for i in range(48,432):
                for e in np.unique(events[i][2]):
                    for h in (1,2,4,8):
                        future[i,int(e),h]=sum((effects.get((j,int(e)),np.zeros(4,dtype=np.int64)) for j in range(i+48,min(432,i+48*h+1),48)),start=np.zeros(4,dtype=np.int64))
            print(json.dumps(dict(batch=batch,stage='P0',status='PASS',independent_events=432,verify=verify)),flush=True)
        lifepath=ROOT/f'B{batch}_{policy}_{"verify" if verify else "primary"}_lifetimes.json.gz'
        with gzip.open(lifepath,'wt') as f:json.dump(life,f,separators=(',',':'))
        if verify:
            original=next(p for p in baseline['points'] if p['policy']==policy)
            assert point=={k:v for k,v in original.items() if k not in ('event_receipt','lifetime_receipt')},policy
        point['event_receipt']=receipt(path);point['lifetime_receipt']=receipt(lifepath)
        result['points'].append(point)
        if not verify:write(basepath,result)
        print(json.dumps({k:point[k] for k in ('batch','policy','expert_h2d_bytes','peer_activation_bytes','fraction_owner_differs_F','fraction_survived_next_demand')}),flush=True)
    assert 'torch' not in sys.modules
    result.update(status='PASS',deterministic_verification=verify,elapsed_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(P/f'B{batch}{"_verify" if verify else ""}.json',result)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--batch',type=int,choices=[8,16,32],required=True);a.add_argument('--verify',action='store_true');args=a.parse_args()
    ROOT.mkdir(exist_ok=True)
    try:run(args.batch,args.verify)
    except BaseException as e:
        write(P/f'failure_B{args.batch}{"_verify" if args.verify else ""}.json',dict(status='FAIL',error=repr(e)));raise
