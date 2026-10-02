#!/usr/bin/env python3
"""Matched IPC/SHM I0 + unchanged trace-only I1. No model entry point."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_rank_oracle_study import memory,process_tree
from trace_comm_input import PACKET,sha
from run_cumem_preflight_retry import snapshot

PACKAGE=PACKET.parents[1]
ROOT=Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/ipc_baseline_20261003')
ORDER=[(0,'T0'),(0,'R3'),(1,'R3'),(1,'T0')]
LABELS={'T0':'T0-IPC (direct P2P / NVSwitch)','R3':'R3-SHM (P2P disabled / host-staged)'}


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def run(label,mode,pass_index=None):
    target=ROOT/label;target.mkdir();smoke=pass_index is None
    env=dict(os.environ,PYTHONPATH=str(PACKAGE),CUDA_VISIBLE_DEVICES='0,1,4,5',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for k in list(env):
        if k.startswith('NCCL_'):del env[k]
    env['NCCL_CUMEM_ENABLE']='0'
    if mode=='R3':env.update(NCCL_P2P_LEVEL='LOC',NCCL_IB_DISABLE='1')
    if smoke:env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(target/'nccl-%h-%p.log'))
    initial=memory();assert initial['host_available_bytes']>=512*2**30 and all(g['used_mib']<1024 for g in initial['gpu'].values()),initial
    if smoke:write(target/'environment.json',snapshot(env))
    worker=PACKAGE/'examples'/('ipc_rebase_smoke.py' if smoke else 'trace_comm_worker.py')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(worker),'--output',str(target),'--mode',mode]
    if not smoke:cmd+=['--counts',str(PACKET/'trace_comm_counts.json'),'--pass-index',str(pass_index),'--ipc-rebase']
    limit=90 if smoke else 180
    state=dict(status='RUNNING',mode=mode,label=LABELS[mode],pass_index=pass_index,smoke=smoke,timeout_seconds=limit,
               command=cmd,transport_env={k:v for k,v in env.items() if k.startswith('NCCL_')},started_unix=time.time(),memory=[initial])
    write(target/'status.json',state);reason=None
    with (target/'run.log').open('w') as log:
        start=time.monotonic();process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=process.pid;next_sample=start+5
        while process.poll() is None:
            remaining=limit-(time.monotonic()-start)
            if remaining<=0:reason='bounded timeout';break
            try:process.wait(timeout=min(1,remaining))
            except subprocess.TimeoutExpired:pass
            if process.poll() is not None:break
            if time.monotonic()>=next_sample and remaining>3:
                sample=memory();sample['group_rss_bytes']=process_tree(process.pid)[1];state['memory'].append(sample);write(target/'status.json',state);next_sample=time.monotonic()+5
                if sample['host_available_bytes']<128*2**30 or sample['group_rss_bytes']>32*2**30 or any(g['free_mib']<8192 for g in sample['gpu'].values()):reason='memory guard';break
        if reason:
            state['signal_elapsed_seconds']=time.monotonic()-start;pids,_=process_tree(process.pid);os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                for pid in pids:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                process.wait(timeout=10)
    state.update(status='PASS' if process.returncode==0 and reason is None else 'FAIL',exit_code=process.returncode,stop_reason=reason,finished_unix=time.time());write(target/'status.json',state)
    assert state['status']=='PASS',str(target/'run.log')
    rs=[json.loads((target/f'rank{r}.json').read_text()) for r in range(4)]
    assert all(r['status']=='PASS' and r['rank']==i and r['payload_valid'] for i,r in enumerate(rs))
    if smoke:
        assert len({r['hostname'] for r in rs})==1 and all(r['world_size']==r['local_world_size']==4 for r in rs)
        paths=[line for p in target.glob('nccl-*.log') for line in p.read_text().splitlines() if 'via ' in line]
        transports=sorted({line.split('via ',1)[1].split()[0] for line in paths})
        assert transports
        if mode=='T0':assert set(transports)=={'P2P/IPC'},transports
        else:assert all(t.startswith('SHM/') for t in transports),transports
        state.update(selected_transports=transports,one_node_four_local_ranks=True,payload_valid_all_ranks=True)
        write(target/'status.json',state)
        (PACKET/f'ipc_rebase_{mode}_transport.log').write_text('\n'.join(paths)+'\n')
    else:
        source=json.loads((PACKET/'trace_comm_counts.json').read_text())
        assert all(r['counts_sha256']==sha(PACKET/'trace_comm_counts.json') and r['ipc_rebase'] and r['warmup_full_traces']==1 and r['timed_full_traces']==3 for r in rs)
        for phase in ('dispatch','combine'):assert sum(r['peer_bytes_per_trace'][phase] for r in rs)==source['peer_bytes_per_trace'][phase]
    print(json.dumps(dict(cell=label,status='PASS',seconds=state['finished_unix']-state['started_unix'])),flush=True)


def main():
    ROOT.mkdir(exist_ok=False);lock=(ROOT/'rebase.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpu_files=['replica_pareto_screen.csv','replica_pareto_screen.json','replica_pareto_events.csv','physical_fkc_schedule_metadata.json']
    cpu_hashes={n:sha(PACKET/n) for n in cpu_files}
    sources=[Path(__file__),PACKAGE/'examples/trace_comm_worker.py',PACKAGE/'examples/ipc_rebase_smoke.py',PACKAGE/'scripts/trace_comm_input.py',PACKET/'trace_comm_counts.json',PACKET/'TRACE_COMM_REPLAY_PROTOCOL.md']
    state=dict(status='RUNNING',plan_commit='29e94fb26c4bb9e91f5e441252b72fdddb335e97',labels=LABELS,completed=[],preflights=[],
               source_sha256={str(p):sha(p) for p in sources},cpu_result_sha256=cpu_hashes,cpu_screen_rerun=False,model_runs=0,
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip())
    try:
        for mode in ('T0','R3'):
            state['active_cell']='I0_'+mode;write(PACKET/'ipc_rebase_progress.json',state)
            run('preflight_'+mode,mode);state['preflights'].append(mode);write(PACKET/'ipc_rebase_progress.json',state)
        for p,mode in ORDER:
            name=f'pass{p}_{mode}';state['active_cell']=name;write(PACKET/'ipc_rebase_progress.json',state)
            run(name,mode,p);state['completed'].append(name);write(PACKET/'ipc_rebase_progress.json',state)
        assert all(sha(PACKET/n)==digest for n,digest in cpu_hashes.items())
        state.update(status='PASS',active_cell=None,cpu_results_unchanged=True)
    except BaseException as exc:state.update(status='FAIL',error=repr(exc));write(PACKET/'ipc_rebase_progress.json',state);raise
    write(PACKET/'ipc_rebase_progress.json',state)


if __name__=='__main__':main()
