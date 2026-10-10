"""Serial best-input batch/length expansion with per-experiment commits/pushes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from pcie_host import ROOT,write

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent
DEST=PKG/'experiments/pcie_topology_ablation_20261009/maxgain_expansion'
BRANCH='codex/pcie-topology-ablation-20261009'
ORDER=((32,64),(64,64),(16,128),(16,256),(32,128),(32,256),(64,128),(64,256))


def main(a):
    a.out.mkdir(parents=True,exist_ok=a.resume);done=[];current=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
             PYTHONPATH=':'.join(map(str,(PKG,PKG/'scripts',PKG/'examples'))),GIT_TERMINAL_PROMPT='0')
    lock=(ROOT/'maxgain_expansion.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.resume:
        previous=json.loads((a.out/'status.json').read_text())
        done=json.loads((a.out/'completed.json').read_text())
        assert previous['status']=='FAIL' and previous['completed']==done
        failed=json.loads((a.out/'step.json').read_text())['step']
        assert failed.endswith('_push') and 'exited 128' in previous['cause'], 'Resume supports a failed publication, never a failed measurement'
        assert failed not in done and not (a.out/'OWNER_INTERRUPTION.json').exists()
        for label in done:
            if label.endswith('_push'):
                assert json.loads((a.out/(label+'_receipt.json')).read_text())['status']=='PASS'
            elif label.endswith(('_nomination','_screen','_final')):
                if label.endswith('_nomination'):job=a.out/label
                else:
                    cell,stage=label.rsplit('_',1);job=a.out/cell/stage
                assert json.loads((job/'status.json').read_text())['status']=='PASS'
                assert json.loads((job/'result.json').read_text())['status']=='PASS'
        recovery=a.out/'recovery'/str(time.time_ns());recovery.mkdir(parents=True)
        for name in ('status.json','failure.json','step.json','completed.json'):
            shutil.copyfile(a.out/name,recovery/name)
        write(recovery/'RESUME.json',dict(status='VALIDATED',failed_step=failed,
            completed_measurements_reused=True,pid=os.getpid(),unix=time.time()))
    def command(name,*args):return [sys.executable,str(PKG/'scripts'/name),*map(str,args)]
    def step(label,argv):
        nonlocal current
        if label in done:return
        if (ROOT/'STOP').exists() or (a.out/'STOP').exists():raise RuntimeError('Owner STOP observed')
        write(a.out/'status.json',dict(status='RUNNING',step=label,pid=os.getpid(),completed=done))
        logfile=a.out/(label+'.log')
        if logfile.exists():logfile=a.out/(label+f'.retry{time.time_ns()}.log')
        with logfile.open('x') as log:
            current=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(a.out/'step.json',dict(step=label,argv=argv,pid=current.pid,started=time.time(),log=str(logfile)))
            code=current.wait();current=None
        if code:raise RuntimeError(f'{label} exited {code}; preserve failure receipts and repair before resuming')
        done.append(label);write(a.out/'completed.json',done)
        print(json.dumps(dict(completed=label)),flush=True)
    def checkpoint(path,message):
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add','--',str(path.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-q','-m',message],cwd=REPO,check=True)
    def push(label):
        if label in done:return
        # Every GPU job has exited, unmapped its source and released the lease
        # before archival/push. Never run Git packing during primary generation.
        step(label,['git','-c','pack.threads=1','-c','pack.window=0','-c','pack.compression=1',
                    'push','origin',f'HEAD:refs/heads/{BRANCH}'])
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        remote=subprocess.check_output(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],cwd=REPO,env=env,text=True).split()[0]
        assert remote==head
        write(a.out/(label+'_receipt.json'),dict(status='PASS',remote_head=remote,local_head=head,unix=time.time(),no_gpu_primary_running=True))
    def subset_archive(source,dest,message):
        if dest.exists():
            assert a.resume
            for name in ('CANDIDATES.json','SELECTION.json'):
                assert (source/name).read_bytes()==(dest/name).read_bytes()
            return
        dest.mkdir(parents=True,exist_ok=False)
        for name in ('CANDIDATES.json','SELECTION.json'):shutil.copyfile(source/name,dest/name)
        checkpoint(dest,message)
    def stop(signum,frame):
        # The GPU supervisor checks ROOT/STOP and cleans up only its own group.
        write(a.out/'OWNER_INTERRUPTION.json',dict(status='OWNER_INTERRUPTED',signal=signum,unix=time.time()))
        write(ROOT/'STOP',dict(reason='Owner interrupted expansion supervisor',pid=os.getpid(),unix=time.time()))
        raise KeyboardInterrupt('Owner interrupted expansion supervisor')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        registration=json.loads((a.prepared/'REGISTRATION.json').read_text());assert registration['status']=='FROZEN'
        saved=DEST/'registration';existing=saved.exists()
        saved.mkdir(parents=True,exist_ok=a.resume)
        if existing:
            assert a.resume and (a.prepared/'REGISTRATION.json').read_bytes()==(saved/'REGISTRATION.json').read_bytes()
        else:shutil.copyfile(a.prepared/'REGISTRATION.json',saved/'REGISTRATION.json')
        for batch,info in registration['batches'].items():
            path=Path(info['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
            if existing:assert path.read_bytes()==(saved/f'B{batch}_CANDIDATES.json').read_bytes()
            else:shutil.copyfile(path,saved/f'B{batch}_CANDIDATES.json')
        if not existing:checkpoint(saved,'Freeze whole-corpus B16 B32 B64 expansion candidates and matched previous-winner controls before physical timing')
        push('registration_push');nominated={}
        for batch,outputs in ORDER:
            cell=f'B{batch}_O{outputs}';cellroot=a.out/cell;cellroot.mkdir(exist_ok=a.resume)
            if batch not in nominated:
                initial=Path(registration['batches'][str(batch)]['path'])
                nomroot=a.out/f'B{batch}_nomination'
                step(f'B{batch}_nomination',command('run_pcie_ours_job.py','--arm','R-NEAR','--out',nomroot,
                     '--search-stage','nomination','--search-spec',initial,'--timeout',14400))
                selected=a.out/f'B{batch}_nomination_selection'
                step(f'B{batch}_nomination_select',command('pcie_maxgain_select.py','--stage','nomination',
                     '--spec',initial,'--root',nomroot,'--out',selected))
                step(f'B{batch}_nomination_archive',command('pcie_maxgain_report.py','--root',nomroot,'--archive','--commit',
                     '--archive-dest',DEST/f'B{batch}_search'))
                subset_archive(selected,DEST/f'B{batch}_search/nomination_selection',f'Freeze B{batch} proxy nominees plus prior-winner control; scores are not serving speedups')
                push(f'B{batch}_nomination_push');nominated[batch]=selected/'CANDIDATES.json'
            spec=json.loads(nominated[batch].read_text());spec['final_output_tokens']=outputs
            spec['cell']=cell;spec['nomination_reused_across_lengths']=True
            frozen=cellroot/'SCREEN_SPEC.json'
            if frozen.exists():assert a.resume and json.loads(frozen.read_text())==spec
            else:write(frozen,spec)
            reg=DEST/cell/'registration'
            if reg.exists():assert a.resume and frozen.read_bytes()==(reg/'SCREEN_SPEC.json').read_bytes()
            else:
                reg.mkdir(parents=True,exist_ok=False)
                shutil.copyfile(frozen,reg/'SCREEN_SPEC.json')
                checkpoint(reg,f'Freeze {cell} independently screened candidate list before requested-length timing')
            screen=cellroot/'screen'
            step(cell+'_screen',command('run_pcie_ours_job.py','--arm','R-NEAR','--out',screen,
                 '--search-stage','screen','--search-spec',frozen,'--timeout',21600))
            selection=cellroot/'screen_selection'
            step(cell+'_screen_select',command('pcie_maxgain_select.py','--stage','screen','--spec',frozen,'--root',screen,'--out',selection))
            step(cell+'_screen_archive',command('pcie_maxgain_report.py','--root',screen,'--archive','--commit','--archive-dest',DEST/cell))
            subset_archive(selection,DEST/cell/'screen_selection',f'Freeze {cell} top3 and required previous-winner control before final repeats')
            push(cell+'_screen_push')
            final=cellroot/'final'
            step(cell+'_final',command('run_pcie_ours_job.py','--arm','R-NEAR','--out',final,
                 '--search-stage','final','--search-spec',selection/'CANDIDATES.json','--timeout',28800))
            step(cell+'_final_archive',command('pcie_maxgain_report.py','--root',final,'--archive','--commit','--archive-dest',DEST/cell))
            report=json.loads((final/'maxgain_validation.json').read_text());assert report['status']=='PASS'
            step(cell+'_matrix_report',command('pcie_maxgain_expansion_report.py','--out',DEST,'--commit'))
            push(cell+'_final_push')
            write(cellroot/'status.json',dict(status='PASS',best_observed=report['best_observed'],primary_repeats=json.loads((final/'result.json').read_text())['primary_repeats']))
        write(a.out/'status.json',dict(status='PASS',pid=os.getpid(),completed=done))
        print(json.dumps(dict(status='PASS',matrix=list(ORDER))),flush=True)
    except BaseException as exc:
        write(a.out/'failure.json',dict(status='FAIL',cause=repr(exc),completed=done,unix=time.time(),child_pid=current.pid if current else None))
        write(a.out/'status.json',dict(status='FAIL',pid=os.getpid(),cause=repr(exc),completed=done))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Recover a failed push after validating and preserving completed measurements')
    main(p.parse_args())
