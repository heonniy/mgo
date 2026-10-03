#!/usr/bin/env python3
"""One-process, hard 8-GiB, checkpointed 264-cell CPU accounting grid."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from cache_policy_cpu import CachePolicyReplay
P=Path(__file__).resolve().parents[1]/'experiments/cache_eviction_substitution_20261003'
ROOT=Path('/home/hwlee/mgo-results/cache_eviction_substitution_20261003')
RHOS=(0,.125,.25,.5,.75)
SUM_FIELDS=('first_copy_fetches reload_fetches replica_fetches total_fetches expert_h2d_bytes peer_activation_bytes dispatch_bytes combine_bytes remote_token_rank_pairs remote_expert_routes raw_expert_routes pre_event_local_exact_hits pre_event_global_hits global_resident_remote_services final_local_services greedy_peer_bytes_saved global_miss_expert_events substituted_sources substituted_routes substituted_gate_mass raw_gate_mass raw_routes protected_exact_misses residual_exact_misses tier_active_exact_hit tier_protected_exact_miss tier_inactive_resident').split()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def summarize(replay,start):
    rows=replay.event_rows[start:]
    out={k:sum(r[k] for r in rows) for k in SUM_FIELDS}
    out.update(events=len(rows),mean_duplicate_slots=float(np.mean([r['duplicate_slots'] for r in rows])),peak_duplicate_slots=max(r['duplicate_peak_in_event'] for r in rows),mean_unique_resident_experts=float(np.mean([r['unique_resident_experts'] for r in rows])),mean_resident_copies=float(np.mean([r['resident_copies'] for r in rows])),rank_evictions=np.sum([r['rank_evictions'] for r in rows],axis=0).tolist())
    out['local_service_fraction']=out['final_local_services']/out['raw_expert_routes']
    out['substituted_route_fraction']=out['substituted_routes']/out['raw_routes']
    out['substituted_gate_mass_fraction']=out['substituted_gate_mass']/out['raw_gate_mass']
    sims=[score for event,score in replay.similarities if event>=start]
    out['accepted_similarity_mean']=float(np.mean(sims)) if sims else None
    out['accepted_similarity_p10']=float(np.quantile(sims,.1)) if sims else None
    lives=[r for r in replay.lifecycle() if r['birth']>=start]
    assert len(lives)==out['replica_fetches']
    durations=[r['end']-r['birth'] for r in lives]
    closed=[r for r in lives if not r['censored']]
    observed48=sum(d>=48 for d in durations)
    unknown48=sum(r['censored'] and r['end']-r['birth']<48 for r in lives)
    out.update(replica_admissions=len(lives),replica_reused=sum(r['reuse_events']>0 for r in lives),replica_evicted=len(closed),replica_reused_before_eviction=sum(r['reuse_events']>0 for r in closed),replica_right_censored=sum(r['censored'] for r in lives),replica_lifetime_p50=float(np.quantile(durations,.5)) if lives else None,replica_lifetime_p90=float(np.quantile(durations,.9)) if lives else None,replica_survived48=observed48,replica_survival48_fraction=observed48/len(lives) if lives else None,replica_survival48_unknown=unknown48,replica_survival48_known_fraction=observed48/(len(lives)-unknown48) if len(lives)>unknown48 else None)
    return out

def cell_id(batch,cache,eviction,sub,rho,fixed):return f'B{batch}_c{round(cache*100)}_{eviction}_s{int(sub)}_'+('fixed460' if fixed is not None else f'rho{rho}')
def run_cell(events,similarity,batch,cache,eviction,sub,rho,fixed):
    slots=int(6144*cache);capacities=[slots//4+(r<slots%4) for r in range(4)]
    replay=CachePolicyReplay(capacities,rho,eviction,similarity,sub,fixed)
    started=time.time()
    for index,item in enumerate(events):
        assert item['event']==index and item['layer']==index%48
        replay.step(item)
    replay.validate()
    assert np.array_equal(replay.copy_count,np.array([[len(replay.owners.get((l,e),())) for e in range(128)] for l in range(48)]))
    result=dict(id=cell_id(batch,cache,eviction,sub,rho,fixed),batch=batch,cache_ratio=cache,capacities=capacities,eviction=eviction,substitution=sub,rho=rho if fixed is None else None,fixed_duplicate_cap=fixed,duplicate_cap=replay.cap,full=summarize(replay,0),decode=summarize(replay,48),final_state_sha256=hashlib.sha256(repr(replay.state()).encode()).hexdigest(),seconds=time.time()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('validate','grid'),required=True);a=parser.parse_args()
    s0=json.loads((P/'S0_capture.json').read_text());assert s0['status']=='PASS'
    assert json.loads((P/'policy_tests.json').read_text())['status']=='PASS'
    for receipt in s0['receipts']:
        if Path(receipt['path']).name.startswith('rank'):assert sha(receipt['path'])==receipt['sha256']
    ready=json.loads((P/'trace_readiness.json').read_text());assert sha(ready['similarity']['path'])==ready['similarity']['sha256']
    similarity=np.load(ready['similarity']['path']);assert similarity.shape==(48,128,128)
    cells=ROOT/'cells';cells.mkdir(exist_ok=True)
    def load(batch):
        r=json.loads((ROOT/f'captures/B{batch}/rank0.json').read_text());assert r['status']=='PASS';return r['events']
    if a.stage=='validate':
        assert not (P/'S1_parity.json').exists(),'No automatic parity retry'
        events=load(8)
        historical=json.loads((P.parent/'fetch_comm_pareto_p2p_20261002/replica_pareto_screen.json').read_text())['points']
        checks=[]
        try:
            for rho,old in zip(RHOS,historical):
                result=run_cell(events,similarity,8,.3,'lru',False,rho,None)
                assert old['rho']==rho
                for scope in ('full','decode'):
                    for key in SUM_FIELDS:
                        if key in old[scope]:assert result[scope][key]==old[scope][key],(rho,scope,key,result[scope][key],old[scope][key])
                    assert result[scope]['mean_unique_resident_experts']==old[scope]['mean_unique_resident_experts']
                assert result['final_state_sha256']==old['final_state_sha256'],(rho,'state hash')
                result['historical_parity']=True;write(cells/(result['id']+'.json'),result)
                checks.append(dict(rho=rho,seconds=result['seconds'],full_decode_counters=True,final_state=True));print(json.dumps(checks[-1]),flush=True)
            write(P/'S1_parity.json',dict(status='PASS',points=checks,reused_in_grid=True,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
        except BaseException as exc:write(P/'S1_parity.json',dict(status='FAIL',points=checks,error=repr(exc)));raise
        return
    assert json.loads((P/'S1_parity.json').read_text())['status']=='PASS'
    matrix=[]
    for batch in (8,32):
        for cache,eviction,sub,rho in itertools.product((.3,.4,.5,.6),('lru','gate','coverage'),(False,True),RHOS):matrix.append((batch,cache,eviction,sub,rho,None))
        if batch==8:
            for cache,eviction,sub in itertools.product((.3,.4,.5,.6),('lru','gate','coverage'),(False,True)):matrix.append((8,cache,eviction,sub,0,460))
    assert len(matrix)==264
    source_hashes={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('cache_policy_cpu.py'))}
    progress_path=P/'grid_progress.json'
    if progress_path.exists():assert json.loads(progress_path.read_text())['source_hashes']==source_hashes,'No mixing implementation versions'
    progress=dict(status='RUNNING',completed=[],total=264,source_hashes=source_hashes,started_unix=time.time())
    events=None;loaded=None
    try:
        for spec in matrix:
            batch,cache,eviction,sub,rho,fixed=spec;ident=cell_id(*spec);path=cells/(ident+'.json')
            if path.exists():result=json.loads(path.read_text())
            else:
                if batch!=loaded:events=load(batch);loaded=batch
                result=run_cell(events,similarity,*spec);write(path,result)
            progress['completed'].append(dict(id=ident,seconds=result['seconds']));progress['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;write(progress_path,progress)
            print(json.dumps(dict(done=len(progress['completed']),total=264,id=ident,seconds=round(result['seconds'],2),peak_rss_mib=progress['peak_rss_mib'])),flush=True)
        progress['status']='PASS'
    except BaseException as exc:progress.update(status='FAIL',error=repr(exc));raise
    finally:progress['finished_unix']=time.time();write(progress_path,progress)
if __name__=='__main__':main()
