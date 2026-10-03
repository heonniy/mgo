#!/usr/bin/env python3
"""Checkpointed 1536-cell two-horizon grid plus 16 frozen seed anchors."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse,fcntl,hashlib,itertools,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
from br_carep_cpu import replay,METRICS,ROW_BYTES,EXPERT_BYTES
ROOT=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
P=Path(__file__).resolve().parents[1]/'experiments/br_ca_carep_cpu_headroom_20261003'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def specs():
    result=[]
    for dataset,horizon,world,batch,cache,eviction,substitution,policy in itertools.product(('MATH','ShareGPT'),(64,256),(4,8),(8,16,32,64),(30,40,50,60),('lru','gate'),(False,True),('BR','CA','CA-rep')):
        row=dict(dataset=dataset,horizon=horizon,world=world,batch=batch,cache=cache,eviction=eviction,substitution=substitution,policy=policy,seed=42,seed_audit=False)
        row['id']=identity(row);result.append(row)
    for dataset,horizon,extreme,seed in itertools.product(('MATH','ShareGPT'),(64,256),(0,1),(7,99)):
        row=dict(dataset=dataset,horizon=horizon,world=4 if extreme==0 else 8,batch=8 if extreme==0 else 64,cache=30 if extreme==0 else 60,eviction='lru' if extreme==0 else 'gate',substitution=bool(extreme),policy='BR',seed=seed,seed_audit=True);row['id']=identity(row);result.append(row)
    assert len(result)==len({r['id'] for r in result})==1552
    return result
def identity(r):return f"{r['dataset']}_h{r['horizon']}_R{r['world']}_B{r['batch']}_c{r['cache']}_{r['eviction']}_s{int(r['substitution'])}_{r['policy']}_seed{r['seed']}"
def summarize(rows,rank_fetch,slots,horizon):
    sums=rows.sum(0);out={key:float(value) for key,value in zip(METRICS,sums)}
    for k in METRICS:
        if 'mass' not in k:out[k]=int(out[k])
    for key in ('resident_copies','unique_resident_experts','duplicate_copies'):
        index=METRICS.index(key);out['mean_'+key]=float(rows[:,index].mean());out['peak_'+key]=int(rows[:,index].max())
    out['max_rank_mandatory_fetches']=int(rows[:,44].max());out['min_rank_mandatory_fetches']=int(rows[:,45].min())
    total=out['first_fetches']+out['reload_fetches']+out['replica_fetches'];out.update(events=len(rows),decode_steps=horizon,total_fetches=total,H2D_bytes=total*EXPERT_BYTES,reload_bytes=out['reload_fetches']*EXPERT_BYTES,peer_bytes=(out['remote_token_rank_pairs']+out['remote_expert_routes'])*ROW_BYTES,dispatch_bytes=out['remote_token_rank_pairs']*ROW_BYTES,return_bytes=out['remote_expert_routes']*ROW_BYTES,cache_turnover_per_decode_step=out['evictions']/slots/horizon,mandatory_rank_fetches=rank_fetch.sum(0).tolist(),max_event_rank_fetch_imbalance=int((rank_fetch.max(1)-rank_fetch.min(1)).max()),cumulative_rank_fetch_imbalance=int(np.ptp(rank_fetch.sum(0))))
    for key in ('exact_global_hits','exact_local_hits','resident_substitute_hits','shared_admission_substitutions','substituted_routes','effective_hits','residual_miss_routes','effective_local_hits'):out[key+'_fraction']=out[key]/out['raw_routes']
    for key in ('exact_global_hit_mass','exact_local_hit_mass','resident_substitute_mass','shared_admission_mass','substituted_mass','effective_hit_mass','residual_miss_mass','effective_local_hit_mass'):out[key+'_fraction']=out[key]/out['raw_gate_mass']
    out['local_service_fraction']=out['local_services']/out['effective_routes'];out['local_service_mass_fraction']=out['local_service_gate_mass']/out['effective_gate_mass'];out['unique_expert_event_miss_fraction']=out['unique_miss_expert_events']/out['unique_raw_expert_events']
    assert out['exact_global_hits']+out['resident_substitute_hits']+out['shared_admission_substitutions']+out['residual_miss_routes']==out['raw_routes']
    return out
def run_cell(spec):
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    started=time.monotonic();dest=ROOT/'cells'/(spec['id']+'.json');assert not dest.exists()
    pack=ROOT/'packed'/spec['dataset']/f"R{spec['world']}_B{spec['batch']}";receipt=json.loads((pack/'receipt.json').read_text());assert receipt['status']=='PASS'
    arrays={key:np.load(pack/(key+'.npy'),mmap_mode='r') for key in ('selected','weights','offsets','prefill_origins','decode_origins','gates')}
    similarity=json.loads((P/'similarity_audit.json').read_text());sim=np.load(similarity['similarity_path']);assert sha(Path(similarity['similarity_path']))==similarity['similarity_sha256']
    slots=48*128*spec['cache']//100;cap=np.array([slots//spec['world']+(r<slots%spec['world']) for r in range(spec['world'])],np.int64)
    result=replay(arrays['selected'],arrays['weights'],arrays['offsets'],arrays['prefill_origins'],arrays['decode_origins'],arrays['gates'],sim,cap,spec['horizon'],spec['eviction']=='gate',spec['substitution'],('BR','CA','CA-rep').index(spec['policy']),spec['seed'])
    rows,fetches=result[:2];assert len(rows)==48*(spec['horizon']+1)
    events=ROOT/'events'/(spec['id']+'.npz');events.parent.mkdir(exist_ok=True);np.savez_compressed(events,rows=rows,rank_fetches=fetches)
    state=hashlib.sha256()
    for array in result[2:]:state.update(array.tobytes())
    out=dict(**spec,status='PASS',global_slots=slots,per_rank_slots=cap.tolist(),full=summarize(rows,fetches,slots,spec['horizon']),decode=summarize(rows[48:],fetches[48:],slots,spec['horizon']),final_state_sha256=state.hexdigest(),alive_replicas=int(result[-1][0]),alive_unused_replicas=int(result[-1][1]),seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,pack_receipt_sha256=sha(pack/'receipt.json'),source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('br_carep_cpu.py'))},event_path=str(events),event_sha256=sha(events))
    out['full']['replicas_never_reused']=out['full']['replicas_evicted_without_reuse']+out['alive_unused_replicas'];assert out['full']['replicas_never_reused']+out['full']['replicas_reused']==out['full']['replica_fetches']
    write(dest,out);print(json.dumps(dict(id=spec['id'],status='PASS',seconds=out['seconds'],peak_rss_mib=out['peak_rss_bytes']/2**20)),flush=True)
def drive(limit,dataset=None):
    lock=(ROOT/'cpu_driver.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((P/'policy_tests.json').read_text())['status']=='PASS'
    tasks=specs();source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('br_carep_cpu.py'))}
    for name in ((dataset,) if dataset else ('MATH','ShareGPT')):
        assert json.loads((P/(name+'_horizon_audit.json')).read_text())['status']=='PASS'
        receipts=json.loads((P/(name+'_pack_receipts.json')).read_text());assert len(receipts)==8
        for receipt in receipts:
            pack=ROOT/'packed'/name/f"R{receipt['world']}_B{receipt['local_batch']}"
            assert all(sha(pack/f['name'])==f['sha256'] for f in receipt['files'])
    done=[];pending=[]
    for spec in tasks:
        path=ROOT/'cells'/(spec['id']+'.json')
        if path.exists():
            c=json.loads(path.read_text());assert c['status']=='PASS' and c['source_hashes']==source_hashes and c['id']==spec['id'];assert sha(Path(c['event_path']))==c['event_sha256'];done.append(dict(id=c['id'],sha256=sha(path),seconds=c['seconds'],peak_rss_bytes=c['peak_rss_bytes']))
        elif dataset is None or spec['dataset']==dataset:pending.append(spec)
    if limit:pending=pending[:limit]
    stopping=[False]
    def stop(*args):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    state=dict(status='RUNNING',planned=1552,main=1536,seed_audit=16,completed=done,source_hashes=source_hashes,max_parallel=16,peak_aggregate_rss_bytes=0)
    write(ROOT/'cpu_progress.json',state);wave=[]
    try:
        for start in range(0,len(pending),16):
            assert psutil.virtual_memory().available>=512*2**30,'Insufficient host memory before wave'
            wave=[]
            for spec in pending[start:start+16]:
                log=ROOT/'cpu_logs'/(spec['id']+'.log');log.parent.mkdir(exist_ok=True);f=log.open('w');p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--cell',json.dumps(spec,separators=(',',':'))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);wave.append((p,f,spec))
            remaining=list(wave)
            while remaining:
                if stopping[0] or (ROOT/'STOP_CPU').exists():raise RuntimeError('Owner stop')
                rss=sum(psutil.Process(p.pid).memory_info().rss for p,_,_ in remaining if p.poll() is None);state['peak_aggregate_rss_bytes']=max(state['peak_aggregate_rss_bytes'],rss)
                if rss>32*2**30 or psutil.virtual_memory().available<256*2**30:raise RuntimeError('CPU aggregate memory guard')
                for entry in list(remaining):
                    p,f,spec=entry
                    if p.poll() is not None:
                        f.close();remaining.remove(entry);assert p.returncode==0,f"Cell failed {spec['id']}, code {p.returncode}"
                        path=ROOT/'cells'/(spec['id']+'.json');c=json.loads(path.read_text());assert c['status']=='PASS';done.append(dict(id=spec['id'],sha256=sha(path),seconds=c['seconds'],peak_rss_bytes=c['peak_rss_bytes']));print(json.dumps(dict(done=len(done),total=1552,id=spec['id'],seconds=round(c['seconds'],2))),flush=True)
                state['updated_unix']=time.time();write(ROOT/'cpu_progress.json',state);time.sleep(2)
        state['status']='PASS' if len(done)==1552 else 'CHECKPOINT';write(ROOT/'cpu_progress.json',state)
    except BaseException as exc:
        state.update(status='STOPPED' if stopping[0] else 'FAILED',reason=repr(exc));write(ROOT/'cpu_progress.json',state);raise
    finally:
        for p,f,spec in wave:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p,f,spec in wave:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            f.close()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cell');parser.add_argument('--limit',type=int,default=0);parser.add_argument('--dataset',choices=['MATH','ShareGPT']);a=parser.parse_args();run_cell(json.loads(a.cell)) if a.cell else drive(a.limit,a.dataset)
