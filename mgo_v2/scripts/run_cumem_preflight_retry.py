#!/usr/bin/env python3
"""Exactly D1 x3 + D2 x1, 90-second limits; never starts E1 or a model."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_rank_oracle_study import memory,process_tree
from trace_comm_input import PACKET,sha

PACKAGE=PACKET.parents[1]
ROOT=Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/cumem_retry_20261003')


def write(path,x):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)


def snapshot(env):
    def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=15)
    versions=json.loads(subprocess.check_output([sys.executable,'-c',
        'import json,torch;print(json.dumps(dict(torch=torch.__version__,torch_path=torch.__file__,cuda=torch.version.cuda,nccl=list(torch.cuda.nccl.version()))))'],text=True,timeout=30))
    return dict(memory=memory(),all_gpu_csv=query(['--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv']),
                compute_process_csv=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv']),
                versions=versions,ambient_nccl_env={k:v for k,v in os.environ.items() if k.startswith('NCCL_')},
                launch_nccl_env={k:v for k,v in env.items() if k.startswith('NCCL_')})


def failure_evidence(target,pid,last_sample):
    pids,_=process_tree(pid);waits=[]
    for p in sorted(pids):
        root=Path('/proc')/str(p)
        try:waits.append(dict(pid=p,wchan=(root/'wchan').read_text().strip(),status=(root/'status').read_text(),cmdline=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode()))
        except OSError:pass
    logs={p.name:p.read_text().splitlines()[-50:] for p in target.glob('nccl-*.log')}
    data=dict(unix=time.time(),process_wait_channels=waits,last_memory_sample=last_sample,last_50_nccl_lines=logs)
    write(target/'stall_evidence.json',data)
    return pids


def run(label,disabled):
    target=ROOT/label;target.mkdir()
    env=dict(os.environ,PYTHONPATH=str(PACKAGE),CUDA_VISIBLE_DEVICES='0,1,4,5',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for key in list(env):
        if key.startswith('NCCL_'):del env[key]
    env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(target/'nccl-%h-%p.log'))
    if disabled:env['NCCL_CUMEM_ENABLE']='0'
    before=snapshot(env);write(target/'environment_before.json',before)
    initial=before['memory'];assert initial['host_available_bytes']>=512*2**30 and all(g['used_mib']<1024 for g in initial['gpu'].values()),initial
    worker=PACKAGE/'examples/cumem_retry_smoke.py'
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(worker),'--output',str(target)]
    if disabled:cmd.append('--cumem-disabled')
    state=dict(status='RUNNING',trial=label,cumem_disabled=disabled,timeout_seconds=90,command=cmd,started_unix=time.time(),memory=[initial])
    write(target/'status.json',state);reason=None
    with (target/'run.log').open('w') as log:
        start=time.monotonic();process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pid']=process.pid;next_sample=start+5
        while process.poll() is None:
            remaining=90-(time.monotonic()-start)
            if remaining<=0:
                reason='90-second timeout';break
            try:process.wait(timeout=min(1,remaining))
            except subprocess.TimeoutExpired:pass
            if process.poll() is not None:break
            if time.monotonic()>=next_sample and 90-(time.monotonic()-start)>3:
                sample=memory();sample['group_rss_bytes']=process_tree(process.pid)[1];state['memory'].append(sample);write(target/'status.json',state);next_sample=time.monotonic()+5
                if sample['host_available_bytes']<128*2**30 or sample['group_rss_bytes']>32*2**30 or any(g['free_mib']<8192 for g in sample['gpu'].values()):reason='memory guard';break
        if reason:
            state['signal_elapsed_seconds']=time.monotonic()-start
            # Snapshot proc/log state immediately, then terminate this group only.
            pids=failure_evidence(target,process.pid,state['memory'][-1])
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                for pid in pids:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                process.wait(timeout=10)
        else:pids=set()
    state.update(status='TIMEOUT' if reason=='90-second timeout' else 'FAIL' if reason or process.returncode!=0 else 'PASS',
                 exit_code=process.returncode,stop_reason=reason,finished_unix=time.time())
    lines=[line for path in target.glob('nccl-*.log') for line in path.read_text().splitlines() if 'via ' in line]
    transports=sorted({s.split('via ',1)[1].split()[0] for s in lines});state['selected_transports']=transports
    state['payload_valid_all_ranks']=False
    if state['status']=='PASS':
        try:
            rs=[json.loads((target/f'rank{r}.json').read_text()) for r in range(4)]
            assert all(r['status']=='PASS' and r['rank']==i and r['payload_valid'] for i,r in enumerate(rs))
            assert any(t.startswith('P2P/') for t in transports)
            if not disabled:assert 'P2P/CUMEM' in transports
            state['payload_valid_all_ranks']=True
        except (AssertionError,OSError,ValueError) as exc:state.update(status='FAIL',validation_error=repr(exc))
    after=memory();state['memory_after_cleanup']=after
    state['target_gpus_released']=all(g['used_mib']<1024 for g in after['gpu'].values())
    write(target/'status.json',state)
    excerpts=[]
    for path in sorted(target.glob('nccl-*.log')):
        logs=path.read_text().splitlines();excerpts+=['# '+path.name]+[s for s in logs if 'NCCL version' in s or 'Init COMPLETE' in s]+[s for s in logs if 'via ' in s][:4]+logs[-50:]+['']
    (PACKET/f'cumem_{label}_nccl.log').write_text('\n'.join(excerpts).rstrip()+'\n')
    write(PACKET/f'cumem_{label}_environment.json',before)
    print(json.dumps({k:state[k] for k in ('trial','status','selected_transports','payload_valid_all_ranks','target_gpus_released')}),flush=True)
    if not state['target_gpus_released']:raise RuntimeError('target GPUs not released; refusing next trial')
    return state


def main():
    ROOT.mkdir(exist_ok=False);lock=(ROOT/'diagnosis.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    progress=dict(status='RUNNING',plan_commit='9d6fea4011c7c5662c109fb57b1a4dac6d4f1d07',trials=[],
                  source_sha256={str(p):sha(p) for p in (Path(__file__),PACKAGE/'examples/cumem_retry_smoke.py')},E1_started=False,model_runs=0)
    try:
        for label,disabled in [('D1_0',False),('D1_1',False),('D1_2',False),('D2',True)]:
            progress['active_trial']=label;write(PACKET/'cumem_retry_progress.json',progress)
            trial=run(label,disabled);progress['trials'].append(trial);write(PACKET/'cumem_retry_progress.json',progress)
        passed=all(t['status']=='PASS' for t in progress['trials'][:3]);alt=progress['trials'][3]['status']=='PASS'
        progress.update(status='COMPLETE',diagnosis=('TRANSIENT_RECOVERED' if alt else 'ALTERNATE_PATH_FAILURE') if passed else ('CUMEM_PATH_UNSTABLE' if alt else 'BROADER_P2P_UVM_FAILURE'),E1_retry_authorized=passed,active_trial=None)
    except BaseException as exc:progress.update(status='BLOCKED',error=repr(exc));write(PACKET/'cumem_retry_progress.json',progress);raise
    write(PACKET/'cumem_retry_progress.json',progress)


if __name__=='__main__':main()
