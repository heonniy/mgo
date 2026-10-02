#!/usr/bin/env python3
"""E1 only. Four fixed cells and two tiny path checks; cannot launch a model."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_rank_oracle_study import memory,group_rss,process_tree
from trace_comm_input import PACKET,sha

PACKAGE=PACKET.parents[1]
ROOT=Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/trace_comm_20261003')
SMOKE=Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/exact_payload_20261003/crossover_worker.py')
ORDER=[(0,'T0'),(0,'R3'),(1,'R3'),(1,'T0')]


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def run(label,mode,pass_index=None):
    target=ROOT/label;target.mkdir(exist_ok=False);smoke=pass_index is None
    env=dict(os.environ,PYTHONPATH=str(PACKAGE),CUDA_VISIBLE_DEVICES='0,1,4,5',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MGO_TRANSPORT=mode)
    for key in list(env):
        if key.startswith('NCCL_'):del env[key]
    if mode=='R3':env.update(NCCL_P2P_LEVEL='LOC',NCCL_IB_DISABLE='1')
    if smoke:env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(target/'nccl-%h-%p.log'))
    initial=memory();assert initial['host_available_bytes']>=512*2**30 and all(g['used_mib']<1024 for g in initial['gpu'].values()),initial
    worker=SMOKE if smoke else PACKAGE/'examples/trace_comm_worker.py'
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(worker),'--output',str(target)]
    command+=['--smoke'] if smoke else ['--counts',str(PACKET/'trace_comm_counts.json'),'--mode',mode,'--pass-index',str(pass_index)]
    state=dict(status='RUNNING',mode=mode,pass_index=pass_index,smoke=smoke,command=command,
               transport_env={k:v for k,v in env.items() if k.startswith('NCCL_')},started_unix=time.time(),memory=[initial])
    write(target/'status.json',state);reason=None
    with (target/'run.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=process.pid
        while process.poll() is None:
            time.sleep(5);sample=memory();sample['group_rss_bytes']=group_rss(process.pid);state['memory'].append(sample);write(target/'status.json',state)
            if time.time()-state['started_unix']>180:reason='bounded timeout'
            if sample['host_available_bytes']<128*2**30 or sample['group_rss_bytes']>32*2**30 or any(g['free_mib']<8192 for g in sample['gpu'].values()):reason='memory guard'
            if reason:
                descendants,_=process_tree(process.pid);os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    for pid in descendants:
                        try:os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    process.wait()
                break
    state.update(status='PASS' if process.returncode==0 and reason is None else 'FAIL',exit_code=process.returncode,stop_reason=reason,finished_unix=time.time());write(target/'status.json',state)
    assert state['status']=='PASS',str(target/'run.log')
    rs=[json.loads((target/f'rank{r}.json').read_text()) for r in range(4)]
    assert all(r['status']=='PASS' for r in rs)
    if smoke:
        assert all(r['results'][0]['payload_valid'] for r in rs)
        lines=[line for path in target.glob('nccl-*.log') for line in path.read_text().splitlines() if 'via ' in line]
        expected='P2P/CUMEM' if mode=='T0' else 'SHM/direct/direct'
        assert any(expected in line for line in lines) and not any('via NET/' in line for line in lines)
        if mode=='R3':assert not any('via P2P/' in line for line in lines)
        (PACKET/f'trace_comm_{mode}_transport.log').write_text('\n'.join(lines)+'\n')
    else:
        counts=json.loads((PACKET/'trace_comm_counts.json').read_text())
        assert all(r['counts_sha256']==sha(PACKET/'trace_comm_counts.json') and r['payload_valid'] and r['warmup_full_traces']==1 and r['timed_full_traces']==3 for r in rs)
        for phase in ('dispatch','combine'):assert sum(r['peer_bytes_per_trace'][phase] for r in rs)==counts['peer_bytes_per_trace'][phase]
    print(json.dumps(dict(cell=label,status='PASS',seconds=state['finished_unix']-state['started_unix'])),flush=True)


def main():
    ROOT.mkdir(exist_ok=False)
    lock=(ROOT/'replay.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths=[Path(__file__),PACKAGE/'examples/trace_comm_worker.py',PACKAGE/'scripts/trace_comm_input.py',PACKET/'trace_comm_counts.json',PACKET/'TRACE_COMM_REPLAY_PROTOCOL.md',SMOKE]
    progress=dict(status='RUNNING',completed=[],preflights=[],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip(),
                  source_sha256={str(p):sha(p) for p in paths},model_runs=0)
    try:
        for mode in ('T0','R3'):
            run('preflight_'+mode,mode);progress['preflights'].append(mode);write(PACKET/'trace_comm_progress.json',progress)
        for pass_index,mode in ORDER:
            label=f'pass{pass_index}_{mode}';progress['active_cell']=label;write(PACKET/'trace_comm_progress.json',progress)
            run(label,mode,pass_index);progress['completed'].append(label);write(PACKET/'trace_comm_progress.json',progress)
        progress.update(status='PASS',active_cell=None)
    except BaseException as exc:
        progress.update(status='FAIL',error=repr(exc));write(PACKET/'trace_comm_progress.json',progress);raise
    write(PACKET/'trace_comm_progress.json',progress)


if __name__=='__main__':main()
