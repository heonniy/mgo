#!/usr/bin/env python3
"""Guarded eight-GPU single-session capture, with no automatic retries."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
PACKAGE=Path(__file__).resolve().parents[1]
P=PACKAGE/'experiments/br_ca_carep_cpu_headroom_20261003'
workers=[];logs=[];stopping=False
def stop(*args):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def write(obj):
    dest=ROOT/'capture_status.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(dest)
def memory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    active={int(x.strip()) for x in apps.splitlines() if x.strip().isdigit()}
    foreign=active-{p.pid for p in workers}
    if foreign:raise RuntimeError('Foreign GPU processes appeared: '+str(sorted(foreign)))
    host=psutil.virtual_memory().available;free={int(i):int(v)*2**20 for i,v in (line.split(',') for line in rows.splitlines())}
    if host<256*2**30 or any(v<8*2**30 for v in free.values()):raise RuntimeError('Memory guard')
    rss=sum(psutil.Process(p.pid).memory_info().rss for p in workers if p.poll() is None)
    return dict(host_available_bytes=host,gpu_free_bytes=free,aggregate_worker_rss_bytes=rss)
def wait_until(predicate,state):
    while not predicate():
        if stopping or (ROOT/'STOP').exists():raise RuntimeError('Owner stop')
        for p in workers:
            if p.poll() is not None:raise RuntimeError(f'Capture worker exited prematurely: {p.pid} code={p.returncode}')
        state.update(resources=memory(),updated_unix=time.time());write(state);time.sleep(5)
def main():
    assert not (ROOT/'capture_status.json').exists(),'No automatic retry or duplicate session'
    assert json.loads((P/'similarity_audit.json').read_text())['status']=='REUSE_VERIFIED'
    assert all((ROOT/(n+'_requests.json')).exists() for n in ('MATH','ShareGPT'))
    snapshot=memory();assert snapshot['host_available_bytes']>=512*2**30 and all(x>=76*2**30 for x in snapshot['gpu_free_bytes'].values())
    state=dict(status='LOADING',model_load_sessions=1,physical_gpus=list(range(8)),replicated_exact_model=True,local_batch=64,cpu_horizons=[64,256],resources=snapshot,started_unix=time.time(),workers=[])
    try:
        for rank in range(8):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            f=(ROOT/f'capture_gpu{rank}.log').open('w');logs.append(f)
            p=subprocess.Popen([sys.executable,'-u',str(PACKAGE/'examples/br_carep_capture.py'),'--rank',str(rank)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);workers.append(p);state['workers'].append(dict(rank=rank,pid=p.pid));write(state)
        wait_until(lambda:all((ROOT/f'gpu{i}_loaded.json').exists() for i in range(8)),state)
        for name in ('MATH','ShareGPT'):
            state['status']='CAPTURING_'+name;write(state);(ROOT/('START_'+name.upper())).touch()
            # ShareGPT workers may exit after writing their final receipt.
            while not all((ROOT/'captures'/name/f'rank{i}/receipt.json').exists() for i in range(8)):
                if stopping or (ROOT/'STOP').exists():raise RuntimeError('Owner stop')
                for i,p in enumerate(workers):
                    if p.poll() is not None and not (ROOT/'captures'/name/f'rank{i}/receipt.json').exists():raise RuntimeError(f'Worker {i} failed, code={p.returncode}')
                state.update(resources=memory(),updated_unix=time.time());write(state);time.sleep(5)
            state['status']='AUDITING_'+name;write(state)
            with (ROOT/(name+'_audit.log')).open('w') as log:
                audit=subprocess.Popen([sys.executable,'-u',str(PACKAGE/'scripts/audit_br_horizons.py'),'--dataset',name],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
                while audit.poll() is None:
                    if stopping:audit.terminate();raise RuntimeError('Owner stop during audit')
                    time.sleep(2)
                assert audit.returncode==0,f'{name} horizon audit failed'
        for p in workers:assert p.wait(timeout=30)==0
        state.update(status='PASS',finished_unix=time.time());write(state)
    except BaseException as exc:
        state.update(status='STOPPED' if stopping else 'FAILED',reason=repr(exc),finished_unix=time.time());write(state);raise
    finally:
        for p in workers:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in workers:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in logs:f.close()
if __name__=='__main__':main()
