#!/usr/bin/env python3
"""C0 exact prices and C1 original five-rho replay freezing."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import argparse,csv,gzip,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
from price_envelope import serial_envelope
P=Path(__file__).resolve().parents[1]/'experiments/synthetic_comm_price_crossover_20261003'
OLD=P.parent/'fetch_comm_pareto_p2p_20261002'
ROOT=Path('/home/hwlee/mgo-results/synthetic_comm_price_crossover_20261003')
RHO=(0,.125,.25,.5,.75)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def receipt(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def csvfile(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
def sources():
    cpu=json.loads((OLD/'replica_pareto_screen.json').read_text());v=json.loads((OLD/'replica_pareto_validation.json').read_text());assert cpu['status']==v['status']=='PASS'
    for name,h in v['output_sha256'].items():assert sha(OLD/name)==h,name
    for r in cpu['provenance']['input_receipts']:assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    commit=cpu['provenance']['source_commit'];historical={}
    for name,h in cpu['provenance']['source_sha256'].items():
        blob=subprocess.check_output(['git','show',f'{commit}:{name}']);assert hashlib.sha256(blob).hexdigest()==h,name
        historical[name]=h
        if name.endswith('/replica_pareto_cpu.py'):
            path=ROOT/'original_replica_pareto_cpu.py';path.write_bytes(blob)
    calibration=P.parent/'hot_expert_replication_threshold_20261003/break_even_microcost.json'
    cv=json.loads((calibration.parent/'validation.json').read_text());assert sha(calibration)==cv['output_sha256'][calibration.name]
    return cpu,dict(historical_source_commit=commit,historical_sources_verified=historical,source_receipts=[receipt(OLD/'replica_pareto_screen.json'),receipt(OLD/'replica_pareto_validation.json'),receipt(calibration)],raw_receipts=cpu['provenance']['input_receipts'],original_policy=receipt(ROOT/'original_replica_pareto_cpu.py'))

def c0():
    assert not (P/'resource_price_sweep.json').exists()
    cpu,provenance=sources();points=[dict(rho=p['rho'],**{k:p['decode'][k] for k in ('total_fetches','expert_h2d_bytes','peer_activation_bytes')}) for p in cpu['points']]
    expected=[(17635,166424739840,334970880),(20862,196878532608,262500352),(24926,235231248384,194297856),(38747,365662568448,50286592),(45886,433034625024,0)]
    assert [(p['total_fetches'],p['expert_h2d_bytes'],p['peer_activation_bytes']) for p in points]==expected
    matrix=json.loads((P/'matrix.json').read_text());rows=[];best=[]
    for price in matrix['lambda_grid']:
        costs=[p['expert_h2d_bytes']+price*p['peer_activation_bytes'] for p in points];value=min(costs);winners=[p['rho'] for p,c in zip(points,costs) if c==value]
        best.append(dict(price_lambda=price,winners=winners,minimum_J_byte=value))
        for p,c in zip(points,costs):rows.append(dict(price_lambda=price,rho=p['rho'],J_byte=c,winner=c==value))
    envelope=serial_envelope([(p['rho'],p['expert_h2d_bytes'],p['peer_activation_bytes']) for p in points])
    result=dict(status='PASS',decision='RESOURCE_PRICE_SHIFT' if len(envelope['regions'])>1 else 'NO_RESOURCE_PRICE_SHIFT',points=points,sweep=rows,winners=best,envelope=envelope,provenance=provenance,units='H2D byte equivalents; lambda is dimensionless resource price, not latency or bandwidth ratio')
    write(P/'resource_price_sweep.json',result);csvfile(P/'resource_price_sweep.csv',rows)
    write(P/'policy_crossover.json',dict(resource=envelope,time=None))
    csvfile(P/'policy_crossover.csv',[dict(model='resource',price_numerator=r['price']['numerator'],price_denominator=r['price']['denominator'],price=r['price']['decimal'],tied_rhos=r['winners']) for r in envelope['crossovers']])
    print(json.dumps(dict(decision=result['decision'],crossovers=envelope['crossovers']),indent=2))

def c1():
    import numpy as np
    from frozen_replica_schedule import FrozenReplicaState,compile_event
    from trace_comm_input import validate_events
    started=time.monotonic();cpu,provenance=sources()
    assert not (P/'frozen_rho_trace_summary.json').exists()
    spec=importlib.util.spec_from_file_location('frozen_original_replica',ROOT/'original_replica_pareto_cpu.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    raw=json.loads(Path(provenance['raw_receipts'][0]['path']).read_text());assert len(raw['events'])==432
    summaries=[]
    for rho in RHO:
        rep=module.ReplicaReplay([461,461,461,460],rho);applied=FrozenReplicaState(rep.capacities,rep.cap)
        events=[];totals={};phase_totals={'prefill':{},'decode':{}}
        path=ROOT/f'schedule_rho{rho}.jsonl.gz'
        with gzip.open(path,'wt') as out:
            for i,e in enumerate(raw['events']):
                assert (e['event'],e['layer'],e['step'])==(i,i%48,i//48)
                o=np.array(e['origin_ranks']);s=np.array(e['raw_selected_experts']);seen=set(rep.seen)
                row,dest,tr,ops=rep.event(e['layer'],o,s)
                event=compile_event(i,e['layer'],o,s,rep,row,dest,tr,ops,seen);applied.apply(event);assert applied.state()==rep.state()
                out.write(json.dumps(event,separators=(',',':'))+'\n')
                for k in ('first_copy_fetches','reload_fetches','replica_fetches','total_fetches','expert_h2d_bytes','peer_activation_bytes','dispatch_bytes','combine_bytes','remote_token_rank_pairs'):
                    totals[k]=totals.get(k,0)+row[k];part=phase_totals['prefill' if i<48 else 'decode'];part[k]=part.get(k,0)+row[k]
                if i>=48:
                    record=dict(source_event=i,step=i//48,layer=i%48,destinations=dest.tolist(),rank_fetch_counts=[sum(c.values()) for c in event['rank_fetch_classes']],cache_sha256=event['post_state_sha256'])
                    assert sum(record['rank_fetch_counts'])==row['total_fetches']
                    for phase,original in (('dispatch','dispatch'),('combine','combine')):
                        matrix=tr[original];record[phase+'_send']=matrix.tolist();record[phase+'_recv']=matrix.T.tolist()
                    events.append(record)
        expected=next(p for p in cpu['points'] if p['rho']==rho)
        assert all(expected['full'][k]==v for k,v in totals.items())
        assert all(expected[phase][k]==v for phase,d in phase_totals.items() for k,v in d.items())
        assert hashlib.sha256(repr(rep.state()).encode()).hexdigest()==expected['final_state_sha256']
        validate_events(events)
        actual=dict(status='PASS',rho=rho,kind='actual',local_batch=8,row_bytes=4096,hidden_size=2048,events=events)
        actualpath=ROOT/f'counts_rho{rho}_actual.json';write(actualpath,actual)
        self_events=[]
        for e in events:
            r={k:e[k] for k in ('source_event','step','layer')}
            for phase in ('dispatch','combine'):
                for direction in ('send','recv'):
                    m=e[phase+'_'+direction];r[phase+'_'+direction]=[[v if a==b else 0 for b,v in enumerate(row)] for a,row in enumerate(m)]
            self_events.append(r)
        selfpath=ROOT/f'counts_rho{rho}_self.json';write(selfpath,dict(status='PASS',rho=rho,kind='self',local_batch=8,row_bytes=4096,hidden_size=2048,actual_sha256=sha(actualpath),events=self_events))
        summary=dict(rho=rho,actual=receipt(actualpath),self=receipt(selfpath),schedule=receipt(path),full=totals,**phase_totals,decode_events=384,validated_action_state_events=432,sum_event_max_rank_fetches=sum(max(e['rank_fetch_counts']) for e in events),final_state_sha256=expected['final_state_sha256'])
        summaries.append(summary);print(json.dumps(dict(stage='C1',rho=rho,decode=summary['decode'])),flush=True)
    assert 'torch' not in sys.modules
    write(P/'frozen_rho_trace_summary.json',dict(status='PASS',schedules=summaries,provenance=provenance,elapsed_seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['c0','c1']);a=parser.parse_args();ROOT.mkdir(exist_ok=True)
    try:(c0 if a.stage=='c0' else c1)()
    except BaseException as exc:write(P/f'failure_{a.stage}.json',dict(status='FAIL',error=repr(exc)));raise
