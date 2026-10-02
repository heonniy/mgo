#!/usr/bin/env python3
"""Minimum owner-authorized controller study, with unchanged resource guards."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from run_rank_oracle_study import PACKAGE, INPUTS, GPUS, NSYS, memory, process_tree, group_rss, write, cell

ROOT=Path('/home/hwlee/mgo-results/controller_overhead_20261002')
OUT=PACKAGE/'experiments/controller_overhead_20261002'
ORIGINAL=PACKAGE/'experiments/rank_demand_oracle_20261002'


def run(label,cells,steps=65,nsys=False):
    target=ROOT/label;target.mkdir(exist_ok=True)
    status=target/'status.json'
    if status.exists():
        old=json.loads(status.read_text())
        if old['status']=='PASS':return
        raise RuntimeError(f'audit previous run before reuse: {target}')
    sample=memory()
    if sample['host_available_bytes']<512*2**30 or any(g['used_mib']>1024 for g in sample['gpu'].values()):
        raise RuntimeError(f'shared-host launch guard: {sample}')
    cells_path=target/'cells.json';write(cells_path,cells)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',PYTHONPATH=str(PACKAGE),OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MOE_EP_DISABLE_ARCHER_EVICT='1',
             MOE_EP_NATIVE_NUMERICS='1',MOE_EP_SLOT_VIEWS='1',MGO_STACK_DUMP_SECONDS='0')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(PACKAGE/'examples/controller_overhead_worker.py'),'--model','/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507',
             '--offload-dir',str(INPUTS/'expert_store'),'--similarity',str(INPUTS/'similarity.npy'),
             '--affinity',str(INPUTS/'affinity.npz'),'--workload',str(INPUTS/'screen_workload.json'),
             '--cells',str(cells_path),'--output',str(target/'receipts'),'--steps',str(steps),'--repeats','1']
    if nsys:
        command=[str(NSYS),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                 '--cuda-event-trace=false','--flush-on-cudaprofilerstop=false',
                 '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(target/'profile')]+command
    state=dict(status='RUNNING',label=label,command=command,cells=cells,steps=steps,started_unix=time.time(),memory=[sample])
    write(status,state)
    with (target/'run.log').open('w') as log:
        process=subprocess.Popen(command,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pid']=process.pid;write(status,state)
        while process.poll() is None:
            time.sleep(10)
            sample=memory();sample['group_rss_bytes']=group_rss(process.pid)
            state['memory'].append(sample);write(status,state)
            if sample['host_available_bytes']<128*2**30 or sample['group_rss_bytes']>320*2**30 or any(g['free_mib']<8192 for g in sample['gpu'].values()):
                descendants,_=process_tree(process.pid)
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    for pid in descendants:
                        try:os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    process.wait()
                state['guard_stop']=sample;break
    state.update(status='PASS' if process.returncode==0 and 'guard_stop' not in state else 'FAIL',exit_code=process.returncode,finished_unix=time.time())
    write(status,state);print(json.dumps({k:state[k] for k in ('label','status','exit_code')}),flush=True)
    if state['status']!='PASS':raise RuntimeError(f'run failed: {target}/run.log')
    if nsys:
        with (target/'export.log').open('w') as log:
            subprocess.run([str(NSYS),'export','--type=sqlite','--output',str(target/'profile.sqlite'),str(target/'profile.nsys-rep')],stdout=log,stderr=subprocess.STDOUT,check=True)


def make_cell(policy,variant,profile=False):
    name=f'b8_{policy}_{variant}'+('_profile' if profile else '')
    result=cell(8,policy,name,'controller',profile=profile)
    result['controller']=variant
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['C0','C1','C2','profiles'],required=True)
    stage=parser.parse_args().stage
    assert json.loads((ORIGINAL/'validation.json').read_text())['status']=='PASS', 'original packet must finish first'
    lock=(ROOT/'study.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Also exclude the original study's GPU launcher.
    original_lock=(Path('/home/hwlee/mgo-results/rank_demand_oracle_20261002')/'study.lock').open('w')
    fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if stage in ('C1','C2'):
        assert json.loads((OUT/'cpu_differential_validation.json').read_text())['status']=='PASS'
    if stage=='C2':
        assert json.loads((ROOT/'control_C1/status.json').read_text())['status']=='PASS'
    if stage=='profiles':
        for v in ('C0','C1','C2'):assert json.loads((ROOT/f'control_{v}/status.json').read_text())['status']=='PASS'
        run('profile_P1',[make_cell('P1','C1',True),make_cell('P1','C2',True)],steps=9,nsys=True)
    else:
        run('control_'+stage,[make_cell(p,stage) for p in ('P0','P1','O0')])

if __name__=='__main__':main()
