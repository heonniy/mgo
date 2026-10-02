#!/usr/bin/env python3
"""Sequential, memory-guarded R4 oracle study. No other user's processes touched."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path('/home/hwlee/mgo-results/rank_demand_oracle_20261002')
INPUTS = Path('/home/hwlee/mgo-results/runtime_validation_20261001')
OUT = PACKAGE / 'experiments/rank_demand_oracle_20261002'
GPUS = [0,1,4,5]
NSYS = PACKAGE.parent / '.build-deps/nsys-2025.6/opt/nvidia/nsight-systems/2025.6.1/target-linux-x64/nsys'


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def memory():
    host = {k: int(v.split()[0])*1024 for k,v in (line.split(':') for line in Path('/proc/meminfo').read_text().splitlines())}
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'], text=True)
    gpu = {int(a): dict(free_mib=int(b),used_mib=int(c)) for a,b,c in (line.split(',') for line in raw.strip().splitlines())}
    return dict(host_available_bytes=host['MemAvailable'], gpu={r:gpu[r] for r in GPUS}, unix=time.time())


def process_tree(root_pid):
    entries = {}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
            entries[int(proc.name)] = (int(fields[1]), int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError): pass
    descendants = {root_pid}
    while True:
        expanded = descendants | {pid for pid,(parent,_) in entries.items() if parent in descendants}
        if expanded == descendants: break
        descendants = expanded
    return descendants, sum(entries.get(pid,(0,0))[1] for pid in descendants)


def group_rss(root_pid):
    return process_tree(root_pid)[1]


def cell(batch, policy, name, mode, planning=None, profile=False):
    return dict(name=name,batch=batch,policy=policy,mode=mode,planning_trace=planning,phase_profile=profile,
                config=dict(global_cache_ratio=.3, substitution_enabled=True, eviction='coverage',
                            same_layer_alpha=.25,path_eta=.5,seed=42,gate_protect_threshold=.20,
                            similarity_threshold=.65,gate_window=128,coverage_k=1,coverage_lambda=2,
                            admission='hungarian_current' if policy=='P1' else 'random'))


def run(label, cells, steps=65, nsys=False):
    target = ROOT/label; target.mkdir(exist_ok=True)
    status_path = target/'status.json'
    if status_path.exists():
        state=json.loads(status_path.read_text())
        if state['status']=='PASS': return
        raise RuntimeError(f'audit previous incomplete run before reuse: {target}')
    sample=memory()
    if sample['host_available_bytes'] < 512*2**30 or any(g['used_mib']>1024 for g in sample['gpu'].values()):
        raise RuntimeError(f'shared-host launch guard: {sample}')
    cells_path=target/'cells.json'; write(cells_path,cells)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',PYTHONPATH=str(PACKAGE),OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MOE_EP_DISABLE_ARCHER_EVICT='1',
             MOE_EP_NATIVE_NUMERICS='1',MOE_EP_SLOT_VIEWS='1',MGO_STACK_DUMP_SECONDS='0')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(PACKAGE/'examples/rank_oracle_worker.py'),'--model','/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507',
             '--offload-dir',str(INPUTS/'expert_store'),'--similarity',str(INPUTS/'similarity.npy'),
             '--affinity',str(INPUTS/'affinity.npz'),'--workload',str(INPUTS/'screen_workload.json'),
             '--cells',str(cells_path),'--output',str(target/'receipts'),'--steps',str(steps),'--repeats','1']
    if nsys:
        command=[str(NSYS),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                 '--cuda-event-trace=false','--flush-on-cudaprofilerstop=false',
                 '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(target/'profile')]+command
    state=dict(status='RUNNING',label=label,command=command,cells=cells,steps=steps,started_unix=time.time(),memory=[sample])
    write(status_path,state)
    with (target/'run.log').open('w') as log:
        process=subprocess.Popen(command,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pid']=process.pid; write(status_path,state)
        while process.poll() is None:
            time.sleep(10)
            sample=memory(); sample['group_rss_bytes']=group_rss(process.pid)
            state['memory'].append(sample); write(status_path,state)
            if sample['host_available_bytes'] < 128*2**30 or sample['group_rss_bytes'] > 320*2**30 or any(g['free_mib']<8192 for g in sample['gpu'].values()):
                descendants, _ = process_tree(process.pid)
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    for pid in descendants:
                        try: os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    process.wait()
                state['guard_stop']=sample
                break
    state.update(status='PASS' if process.returncode==0 else 'FAIL',exit_code=process.returncode,finished_unix=time.time())
    write(status_path,state)
    print(json.dumps(dict(label=label,status=state['status'],seconds=state['finished_unix']-state['started_unix'])),flush=True)
    if state['status']!='PASS': raise RuntimeError(f'run failed: {target}/run.log')
    if nsys:
        with (target/'export.log').open('w') as log:
            subprocess.run([str(NSYS),'export','--type=sqlite','--output',str(target/'profile.sqlite'),str(target/'profile.nsys-rep')],stdout=log,stderr=subprocess.STDOUT,check=True)


def trace(label,name): return str(ROOT/label/'receipts'/f'{name}-evidence-rank{{rank}}.json')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['smoke','planning','timing','profiles'],required=True)
    args=parser.parse_args(); ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'study.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=='smoke':
        run('smoke_plan',[cell(4,'O0','smoke_plan','planning')],steps=2)
        for p in ('P0','P1','O0'):
            for enabled in (False,True):
                name=f'smoke_{p}_{int(enabled)}'
                run(name,[cell(4,p,name,'replay',trace('smoke_plan','smoke_plan'),enabled)],steps=2)
        for p in ('P0','P1','O0'):
            for r in range(4):
                a=json.loads((ROOT/f'smoke_{p}_0/receipts/smoke_{p}_0-evidence-rank{r}.json').read_text())
                b=json.loads((ROOT/f'smoke_{p}_1/receipts/smoke_{p}_1-evidence-rank{r}.json').read_text())
                assert a==b, (p,r,'instrumentation parity')
        write(OUT/'smoke_validation.json',dict(status='PASS',policies=3,ranks=4,forwards=2,profile_on_off_exact=True))
    elif args.stage=='planning':
        assert json.loads((OUT/'smoke_validation.json').read_text())['status']=='PASS'
        for batch in (4,8,16):
            name=f'plan_b{batch}'; run(name,[cell(batch,'O0',name,'planning')])
    elif args.stage=='timing':
        for batch in (4,8,16):
            plan=f'plan_b{batch}'
            assert json.loads((ROOT/plan/'status.json').read_text())['status']=='PASS'
            cells=[]
            for repeat,order in enumerate(itertools.permutations(('P0','P1','O0'))):
                cells.extend(cell(batch,p,f'b{batch}_{p}_rep{repeat}','replay',trace(plan,plan)) for p in order)
            run(f'timing_b{batch}',cells)
    else:
        for batch in (4,8,16):
            assert json.loads((ROOT/f'timing_b{batch}/status.json').read_text())['status']=='PASS'
        for p in ('P0','P1','O0'):
            name=f'profile_b8_{p}'; run(name,[cell(8,p,name,'replay',trace('plan_b8','plan_b8'),True)],nsys=True)

if __name__=='__main__': main()
