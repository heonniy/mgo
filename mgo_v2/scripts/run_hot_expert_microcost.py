#!/usr/bin/env python3
"""Guarded GPU0/1/4/5 handoff for the bounded model-free H1 calibration."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import batch_comm_common as common
from run_rank_oracle_study import write
from trace_comm_input import sha
P=Path(__file__).resolve().parents[1]/'experiments/hot_expert_replication_threshold_20261003'
ROOT=Path('/home/hwlee/mgo-results/hot_expert_replication_threshold_20261003/H1')
TARGETS={0,1,4,5}

def apps():
    mapping={u.strip():int(g) for g,u in (s.split(',') for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    return [(mapping[u.strip()],int(pid)) for u,pid in (s.split(',') for s in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())]

def restore(original,paused):
    current=json.loads((common.LOAD/'processes.json').read_text());restored=[]
    assert not (common.LOAD/'STOP').exists(),'External STOP requested; do not override'
    for w in paused:
        if common.owned(w['pid']):continue
        g=w['gpu']
        if any(g==busy for busy,_ in apps()):
            restored.append(dict(gpu=g,status='SKIPPED_BUSY'));continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (common.LOAD/f'gpu{g}.log').open('a') as log:
            proc=subprocess.Popen([common.PYTHON,'-u',str(common.WORKER)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        current=[r for r in current if r['gpu']!=g]+[dict(gpu=g,pid=proc.pid)]
        restored.append(dict(gpu=g,pid=proc.pid,status='STARTED'))
        write(common.LOAD/'processes.json',current)
    assert all(r['gpu'] in TARGETS for r in restored)
    return restored

def main(resume_startup=False):
    assert json.loads((P.parent/'future_rank_affinity_placement_20261003/validation.json').read_text())['status']=='PASS'
    assert json.loads((P/'H0.json').read_text())['status']=='PASS'
    previous=None
    if resume_startup:
        previous=json.loads((P/'H1_startup_failure.json').read_text())
        assert previous['status']=='FAIL' and [r['label'] for r in previous['cells']]==['smoke_T0','smoke_R3']
        failed=ROOT/'pair_pass0_T0'
        assert not list(failed.glob('rank*.json')) and "NCCL_P2P_DISABLE" in (failed/'run.log').read_text()
        assert not (ROOT/'pair_pass0_T0_corrected').exists(),'Startup correction already attempted'
    else:
        assert not ROOT.exists(),'Do not automatically repeat H1'
        ROOT.mkdir()
    common.ROOT=ROOT;common.PACKET=P
    original=json.loads((common.LOAD/'processes.json').read_text())
    paused=[r for r in original if r['gpu'] in TARGETS and common.owned(r['pid'])]
    ours={r['pid'] for r in paused}
    assert all(pid in ours for g,pid in apps() if g in TARGETS),'Foreign target-GPU job; do not touch'
    state=dict(status='RUNNING',workers_before=original,paused_workers=paused,started_unix=time.time(),cells=list(previous['cells']) if previous else [],startup_correction=resume_startup)
    write(P/'H1.json',state)
    try:
        for r in paused:os.kill(r['pid'],signal.SIGTERM)
        deadline=time.monotonic()+20
        while any(common.owned(r['pid']) for r in paused) and time.monotonic()<deadline:time.sleep(.25)
        for r in paused:
            if common.owned(r['pid']):os.kill(r['pid'],signal.SIGKILL)
        deadline=time.monotonic()+20
        while any(g in TARGETS for g,_ in apps()) and time.monotonic()<deadline:time.sleep(.5)
        assert not any(g in TARGETS for g,_ in apps())
        for mode in (() if resume_startup else ('T0','R3')):
            label='smoke_'+mode
            common.run(label,'ipc_rebase_smoke.py',['--mode',mode],mode=mode,smoke=True)
            state['cells'].append(dict(label=label,kind='smoke',mode=mode));write(P/'H1.json',state)
        for index,(pass_id,mode) in enumerate(((0,'T0'),(0,'R3'),(1,'R3'),(1,'T0'))):
            label=f'pair_pass{pass_id}_{mode}'
            if resume_startup and index==0:label+='_corrected'
            common.run(label,'hot_expert_microcost.py',['--mode',mode,'--kind','pair'],mode=mode)
            state['cells'].append(dict(label=label,kind='pair',mode=mode,pass_id=pass_id));write(P/'H1.json',state)
        common.run('h2d','hot_expert_microcost.py',['--mode','T0','--kind','h2d'],mode='T0')
        state['cells'].append(dict(label='h2d',kind='h2d',mode='T0'));state['status']='PASS'
    except BaseException as exc:
        state.update(status='FAIL',error=repr(exc));raise
    finally:
        try:state['restored_workers']=restore(original,paused)
        except BaseException as exc:
            state['restore_error']=repr(exc);state['status']='FAIL';raise
        finally:
            state['finished_unix']=time.time()
            state['receipts']=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in ROOT.glob('*/*') if f.is_file()]
            write(P/'H1.json',state)
            print(json.dumps({k:state.get(k) for k in ('status','error','restore_error','restored_workers')}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume-startup',action='store_true');a=parser.parse_args();main(a.resume_startup)
