#!/usr/bin/env python3
"""Frozen 120 cells, with all 24 N0 references checked before comparisons."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[var]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
import argparse,hashlib,itertools,json,subprocess,time
from pathlib import Path
import numpy as np
from dynamic_refresh_cpu import DynamicReplay,FutureDemand
from run_cache_policy_grid import summarize,write,sha
P=Path(__file__).resolve().parents[1]/'experiments/dynamic_replica_refresh_20261003'
OLD=P.parent/'cache_eviction_substitution_20261003'
ROOT=Path('/home/hwlee/mgo-results/dynamic_replica_refresh_20261003')
OLDROOT=Path('/home/hwlee/mgo-results/cache_eviction_substitution_20261003')
SOURCE='d66641405f2aaa7ebeb28659aebc3511bd6ca444'
POLICIES=('N0','C1','C2','O4','OR')

def specs():
    return [(b,c,e,s,r) for b,cs in ((32,(.4,.6)),(8,(.6,))) for c,e,s,r in itertools.product(cs,('lru','gate'),(False,True),(.125,.25))]
def ident(spec,policy):
    b,c,e,s,r=spec;return f'B{b}_c{round(c*100)}_{e}_s{int(s)}_rho{r}_{policy}'
def source_id(spec):return ident(spec,'N0').removesuffix('_N0')

def audit():
    ready=json.loads((OLD/'trace_readiness.json').read_text());s0=json.loads((OLD/'S0_capture.json').read_text());assert s0['status']=='PASS'
    receipts=[]
    for r in s0['receipts']:
        if Path(r['path']).name.startswith('rank'):
            assert sha(r['path'])==r['sha256'];receipts.append(r)
    assert len(receipts)==8 and sha(ready['similarity']['path'])==ready['similarity']['sha256']
    source_files=[]
    repo=P.parents[2]
    for name in ('mgo_v2/scripts/cache_policy_cpu.py','mgo_v2/scripts/replica_pareto_cpu.py','mgo_v2/scripts/run_cache_policy_grid.py','mgo_v2/mgo_v2/substitution.py','mgo_v2/mgo_v2/types.py'):
        old=subprocess.check_output(['git','show',f'{SOURCE}:{name}'],cwd=repo)
        current=repo/name;expected=hashlib.sha256(old).hexdigest();assert sha(current)==expected
        source_files.append(dict(path=str(current),sha256=expected))
    recorded={Path(r['path']).stem:r for r in json.loads((OLD/'cell_receipts.json').read_text())}
    refs=[]
    for spec in specs():
        r=recorded[source_id(spec)];assert sha(r['path'])==r['sha256'];refs.append(r)
    # Cache30 is descriptive historical context for CACHE_RELIEF_PRESERVED.
    context=[]
    for b,e,s,r in itertools.product((8,32),('lru','gate'),(False,True),(.125,.25)):
        item=recorded[source_id((b,.3,e,s,r))];assert sha(item['path'])==item['sha256'];context.append(item)
    out=dict(status='PASS',source_commit=SOURCE,traces=receipts,similarity=ready['similarity'],source_files=source_files,N0_references=refs,historical_cache30=context)
    write(P/'source_verification.json',out);return out

def load_events(batch):
    raw=json.loads((OLDROOT/f'captures/B{batch}/rank0.json').read_text());assert raw['status']=='PASS'
    out=[]
    for r in raw['events']:
        out.append(dict(event=r['event'],step=r['step'],layer=r['layer'],origin_ranks=np.asarray(r['origin_ranks'],dtype=np.int64),raw_selected_experts=np.asarray(r['raw_selected_experts'],dtype=np.int64),routing_weights=np.asarray(r['routing_weights'],dtype=np.float32),gate_scores=np.asarray(r['gate_scores'],dtype=np.float32)))
    return out

def extended(replay,start):
    rows=replay.event_rows[start:];out=summarize(replay,start)
    for k in ('refresh_admissions','refresh_duplicate_evictions','refresh_immediate_peer_saved','refresh_predicted_horizon_peer_saved','pre_refresh_peer_activation_bytes'):out[k]=sum(r[k] for r in rows)
    assert out['pre_refresh_peer_activation_bytes']-out['peer_activation_bytes']==out['refresh_immediate_peer_saved']
    rs=[r for r in replay.refresh_records if r['event']>=start]
    for field in ('victim_copy_age','victim_idle_age'):
        vals=[r[field] for r in rs]
        out[field+'_p50']=float(np.quantile(vals,.5)) if vals else None;out[field+'_p90']=float(np.quantile(vals,.9)) if vals else None
    out['refresh_primary_victims']=sum(r['victim_primary'] for r in rs)
    out['duplicates_never_reused']=out['replica_admissions']-out['replica_reused']
    out['duplicates_never_reused_fraction']=out['duplicates_never_reused']/out['replica_admissions'] if out['replica_admissions'] else None
    return out

def run(events,sim,spec,policy):
    b,c,e,s,r=spec;slots=int(6144*c);caps=[slots//4+(i<slots%4) for i in range(4)]
    future=FutureDemand(events) if policy in ('O4','OR') else None
    replay=DynamicReplay(caps,r,e,sim,s,policy,future);started=time.time()
    for i,item in enumerate(events):
        assert item['event']==i and item['layer']==i%48
        replay.step(item)
    replay.validate()
    out=dict(id=ident(spec,policy),batch=b,cache_ratio=c,eviction=e,substitution=s,rho=r,policy=policy,duplicate_cap=replay.cap,full=extended(replay,0),decode=extended(replay,48),final_state_sha256=hashlib.sha256(repr(replay.state()).encode()).hexdigest(),seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    # Exact matched trajectory decomposition, with no per-swap causal claim.
    reference=json.loads((OLDROOT/'cells'/(source_id(spec)+'.json')).read_text()) if policy=='N0' else json.loads((ROOT/'cells'/(ident(spec,'N0')+'.json')).read_text())
    for scope in ('full','decode'):
        a=out[scope];base=reference[scope]
        a['net_peer_saved_vs_N0']=base['peer_activation_bytes']-a['peer_activation_bytes']
        a['realized_downstream_peer_saved']=a['net_peer_saved_vs_N0']-a['refresh_immediate_peer_saved']
        a['delta_H2D_bytes']=a['expert_h2d_bytes']-base['expert_h2d_bytes'];a['delta_peer_bytes']=-a['net_peer_saved_vs_N0']
        a['delta_reload']=a['reload_fetches']-base['reload_fetches'];a['delta_local_service']=a['local_service_fraction']-base['local_service_fraction'];a['delta_unique_coverage']=a['mean_unique_resident_experts']-base['mean_unique_resident_experts']
    if policy=='N0':
        for scope in ('full','decode'):
            for k,v in reference[scope].items():assert out[scope][k]==v,(out['id'],scope,k,out[scope][k],v)
        assert out['final_state_sha256']==reference['final_state_sha256'],out['id']
        out['N0_reference_parity']=True
    write(ROOT/'swaps'/(out['id']+'.json'),replay.refresh_records)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('audit','N0','refresh'),required=True);a=p.parse_args()
    ROOT.mkdir(exist_ok=True);(ROOT/'cells').mkdir(exist_ok=True);(ROOT/'swaps').mkdir(exist_ok=True)
    if a.stage=='audit':audit();return
    verify=json.loads((P/'source_verification.json').read_text());assert verify['status']=='PASS'
    assert json.loads((P/'policy_tests.json').read_text())['status']=='PASS'
    for receipt in verify['traces']+verify['source_files']+verify['N0_references']+verify['historical_cache30']+[verify['similarity']]:assert sha(receipt['path'])==receipt['sha256']
    sim=np.load(verify['similarity']['path'])
    if a.stage=='refresh':assert json.loads((P/'N0_validation.json').read_text())['status']=='PASS'
    path=P/('N0_validation.json' if a.stage=='N0' else 'refresh_progress.json')
    assert not path.exists(),'No automatic experiment retry'
    sources={str(f):sha(f) for f in (Path(__file__),Path(__file__).with_name('dynamic_refresh_cpu.py'))}
    state=dict(status='RUNNING',source_hashes=sources,cells=[],started_unix=time.time(),expected_cells=24 if a.stage=='N0' else 96)
    events=None;batch=None
    try:
        for spec in specs():
            if batch!=spec[0]:events=load_events(spec[0]);batch=spec[0]
            for policy in (('N0',) if a.stage=='N0' else POLICIES[1:]):
                result=run(events,sim,spec,policy);write(ROOT/'cells'/(result['id']+'.json'),result)
                state['cells'].append(dict(id=result['id'],seconds=result['seconds'],sha256=sha(ROOT/'cells'/(result['id']+'.json'))));state['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;write(path,state)
                print(json.dumps(dict(done=len(state['cells']),total=state['expected_cells'],id=result['id'],seconds=round(result['seconds'],2),peak_rss_mib=state['peak_rss_mib'])),flush=True)
        assert len(state['cells'])==state['expected_cells'];state['status']='PASS'
    except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
    finally:state['finished_unix']=time.time();write(path,state)
if __name__=='__main__':main()
