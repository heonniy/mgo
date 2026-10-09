"""Ordered CPU/GPU best64 pipeline; each GPU job owns the normal burn lease."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pcie_host import ROOT,write

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent
DEST=PKG/'experiments/pcie_topology_ablation_20261009/maxgain64'


def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
             PYTHONPATH=':'.join([str(PKG),str(PKG/'scripts'),str(PKG/'examples')]))
    completed=[];current=None
    def step(label,command):
        nonlocal current
        if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP observed before step')
        write(a.out/'status.json',dict(status='RUNNING',step=label,pid=os.getpid(),completed=completed))
        with (a.out/(label+'.log')).open('w') as log:
            current=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(a.out/'step.json',dict(step=label,argv=command,pid=current.pid,started=time.time()))
            code=current.wait();current=None
        if code:raise RuntimeError(f'{label} exited {code}; see its log and preserved child failure receipts')
        completed.append(label);write(a.out/'completed.json',completed)
        print(json.dumps(dict(completed=label)),flush=True)
    py=sys.executable
    def script(name):return str(PKG/'scripts'/name)
    def commit(path,message):
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add','--',str(path.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m',message],cwd=REPO,check=True)
    try:
        write(a.out/'status.json',dict(status='RUNNING',step='wait_corpus',pid=os.getpid(),completed=[]))
        started=time.monotonic()
        while True:
            if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP observed')
            status_path=a.cpu_job/'status.json'
            if status_path.exists():
                status=json.loads(status_path.read_text())
                if status['status']=='FAIL':raise RuntimeError('Whole-corpus CPU job failed; preserve and repair before GPU search')
                if status['status']=='PASS':
                    try:os.kill(status['pid'],0)
                    except ProcessLookupError:break
            if time.monotonic()-started>21600:raise TimeoutError('Corpus preparation did not finish within 6 hours')
            time.sleep(1)
        candidates=a.out/'candidates'
        step('prepare_candidates',[py,script('prepare_pcie_maxgain_candidates.py'),'--pool',str(a.pool),'--out',str(candidates)])
        registration=DEST/'registration';registration.mkdir(parents=True,exist_ok=False)
        for source in (a.pool/'POOL.json',candidates/'CANDIDATES.json'):
            (registration/source.name).write_bytes(source.read_bytes())
        write(registration/'PREFLIGHT.json',json.loads((ROOT/'maxgain_replay_preflight.json').read_text()))
        commit(registration,'Census entire frozen ShareGPT corpus and freeze 33 real batch64 candidates before physical nomination')
        spec=candidates/'CANDIDATES.json'
        for stage in ('nomination','screen','final'):
            root=a.out/stage
            step(stage,[py,script('run_pcie_ours_job.py'),'--arm','R-NEAR','--out',str(root),
                       '--search-stage',stage,'--search-spec',str(spec),'--timeout','10800'])
            if stage!='final':
                selection=a.out/(stage+'_selection')
                step(stage+'_selection',[py,script('pcie_maxgain_select.py'),'--stage',stage,'--spec',str(spec),
                          '--root',str(root),'--out',str(selection)])
                spec=selection/'CANDIDATES.json'
                saved=DEST/(stage+'_selection');saved.mkdir(parents=True,exist_ok=False)
                for name in ('CANDIDATES.json','SELECTION.json'):(saved/name).write_bytes((selection/name).read_bytes())
                commit(saved,f'Freeze best64 {stage} selection before subsequent measurements; retain every candidate score')
            step(stage+'_archive',[py,script('pcie_maxgain_report.py'),'--root',str(root),'--archive','--commit'])
        result=json.loads((a.out/'final/maxgain_validation.json').read_text())
        assert result['status']=='PASS'
        write(a.out/'result.json',dict(status='PASS',best_observed=result['best_observed'],completed=completed,
             scope=result['scope'],original_full_plan_goal_still_active=True))
        write(a.out/'status.json',dict(status='PASS',pid=os.getpid(),completed=completed))
        print(json.dumps(dict(status='PASS',best_observed=result['best_observed'])),flush=True)
    except BaseException as exc:
        write(a.out/'failure.json',dict(status='FAIL',cause=repr(exc),completed=completed,unix=time.time(),
              child_pid=current.pid if current else None))
        write(a.out/'status.json',dict(status='FAIL',cause=repr(exc),pid=os.getpid(),completed=completed))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pool',type=Path,required=True);p.add_argument('--cpu-job',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
