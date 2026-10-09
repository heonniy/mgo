"""Queue all three native baselines AFTER the complete best-input expansion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from pcie_host import ROOT,write,available_bytes
from pcie_gpu_occupancy import process_identity
from pcie_receipts import read_receipt
from continue_pcie_deepspeed import validate_smoke

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent
EXP=PKG/'experiments/pcie_topology_ablation_20261009'
BRANCH='codex/pcie-topology-ablation-20261009'


def validate_native_smoke(root,system):
    if system=='deepspeed':return validate_smoke(root)
    result=read_receipt(root/'result.json')
    assert read_receipt(root/'status.json')['status']=='PASS' and result['status']=='PASS' and result['smoke']
    for repeat in (0,1):
        row=read_receipt(root/f'repeat{repeat}.json')
        assert row['status']=='PASS' and row['smoke'] and row['output_tokens']==2
        assert len(row['tokens'])==4 and all(len(t)==2 for t in row['tokens'])
        if system=='infinity':
            assert row['kv_released'] and row['eam_calls']==96
            assert row['cache_after']['peak_accounted_bytes']<=17392730112
            assert row['pinned_host_peak_bytes']>=row['pinned_host_bytes']>=0
        else:
            assert row['finite_logits'] and row['synchronous_batch'] and row['offload_kqv']
            assert row['expert_placement']=='balanced3' and row['cpu_threads']==32
    write(root/'SMOKE_VALIDATION.json',dict(status='PASS',system=system,
        scope='Native progress/numerics/budget smoke only; full64 still required'))


def main(a):
    a.out.mkdir(parents=True,exist_ok=False);done=[];current=None;current_cpu=False
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
        PYTHONPATH=':'.join(map(str,(PKG,PKG/'scripts',PKG/'examples'))),GIT_TERMINAL_PROMPT='0')
    def check_stop():
        if (ROOT/'STOP').exists() or (a.out/'STOP').exists():
            write(ROOT/'STOP',dict(reason='Owner stopped baseline continuation',pid=os.getpid(),unix=time.time()))
            raise RuntimeError('Owner STOP observed')
    def state(step):write(a.out/'status.json',dict(status='RUNNING',step=step,pid=os.getpid(),completed=done))
    def command(name,*args):return [sys.executable,str(PKG/'scripts'/name),*map(str,args)]
    def run(step,argv,cpu=False):
        nonlocal current,current_cpu
        check_stop();state(step);quiet=None
        if cpu:
            quiet=(ROOT/'gpu_experiment.lock').open('w')
            while True:
                check_stop()
                try:fcntl.flock(quiet,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(2)
        try:
            with (a.out/(step+'.log')).open('w') as log:
                current_cpu=cpu
                current=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=cpu)
                write(a.out/'step.json',dict(step=step,pid=current.pid,argv=argv,cpu_only=cpu,started_unix=time.time()))
                started=time.monotonic()
                while current.poll() is None:
                    check_stop()
                    if cpu and (available_bytes()<128*2**30 or time.monotonic()-started>21600):
                        os.killpg(current.pid,signal.SIGTERM)
                        try:current.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(current.pid,signal.SIGKILL);current.wait()
                        raise RuntimeError('CPU setup host-memory or six-hour bound failed; attempt preserved')
                    time.sleep(1)
                code=current.returncode;current=None
            if code:raise RuntimeError(f'{step} exited {code}; inspect preserved log and repair')
            done.append(step);write(a.out/'completed.json',done);print(json.dumps(dict(completed=step)),flush=True)
        finally:
            if quiet:quiet.close()
    def checkpoint(path,message):
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        subprocess.run(['git','add','--',str(path.relative_to(REPO))],cwd=REPO,check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m',message],cwd=REPO,check=True)
    def push(label):
        run(label,['git','-c','pack.threads=1','-c','pack.window=0','-c','pack.compression=1',
                   'push','origin',f'HEAD:refs/heads/{BRANCH}'],cpu=True)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        remote=subprocess.check_output(['git','ls-remote','origin',f'refs/heads/{BRANCH}'],cwd=REPO,env=env,text=True).split()[0]
        assert remote==head;write(a.out/(label+'_receipt.json'),dict(status='PASS',local_head=head,remote_head=remote,unix=time.time()))
    def saved_receipt(name,message):
        path=EXP/'baseline_builds'/name
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
        checkpoint(path,message)
    def owner_interrupt(signum,frame):
        write(ROOT/'STOP',dict(reason='Owner interrupted baseline continuation',pid=os.getpid(),unix=time.time()))
        write(a.out/'OWNER_INTERRUPTION.json',dict(status='OWNER_INTERRUPTED',signal=signum,unix=time.time()))
        raise KeyboardInterrupt('Owner interrupted baseline continuation')
    signal.signal(signal.SIGTERM,owner_interrupt);signal.signal(signal.SIGINT,owner_interrupt)
    try:
        state('wait_all_eight_expansion_cells')
        argv=(Path('/proc')/str(a.predecessor_pid)/'cmdline').read_bytes().split(b'\0') if process_identity(a.predecessor_pid)==a.predecessor_start_ticks else []
        if argv:
            assert any(b'run_pcie_maxgain_expansion.py' in v for v in argv) and str(a.predecessor).encode() in argv
        write(a.out/'predecessor_identity.json',dict(pid=a.predecessor_pid,start_ticks=a.predecessor_start_ticks,argv=[v.decode() for v in argv if v],root=str(a.predecessor)))
        started=time.monotonic()
        while True:
            check_stop();status=read_receipt(a.predecessor/'status.json')
            if status['status']=='FAIL':raise RuntimeError('Expansion failed; do not bypass its completion gate')
            alive=process_identity(a.predecessor_pid)==a.predecessor_start_ticks
            if not alive:
                assert status['status']=='PASS','Expansion exited before all cells were validated and pushed'
                break
            if time.monotonic()-started>7*86400:raise TimeoutError('Expansion wait exceeded seven days')
            time.sleep(2)
        matrix=read_receipt(EXP/'maxgain_expansion/MATRIX_RESULTS.json')
        assert matrix['status']=='PASS' and len(matrix['cells'])==9
        assert all(c['status'] in ('PASS','HISTORICAL_PASS') for c in matrix['cells'])
        for b in (16,32,64):
            for n in (64,128,256):
                if (b,n)==(16,64):continue
                assert read_receipt(a.predecessor/f'B{b}_O{n}/status.json')['status']=='PASS'
        assert read_receipt(a.predecessor/'B64_O256_final_push_receipt.json')['status']=='PASS'
        ours=read_receipt(EXP/'stage2_grouped/cohort/live_cohort_validation.json')
        assert ours['status']=='PASS' and len(ours['arms'])==7
        done.append('expansion_terminal_validated_and_pushed')
        write(a.out/'PREREQUISITES.json',dict(status='PASS',all_eight_cells=True,baseline_cell='R4_C30_B16_L512_O64',proposed='G-NEAR',
            source='Original frozen ShareGPT64, not selected-input search',output_tokens=64,primary_repeats_per_baseline=5))
        for system in ('deepspeed','infinity','llama'):
            cell=a.out/system;cell.mkdir()
            if system=='deepspeed':
                run('deepspeed_cpu_preflight',command('pcie_deepspeed_leaf_preflight.py','--out',cell/'leaf_preflight.json'),cpu=True)
            elif system=='infinity':
                run('infinity_native_build',command('build_pcie_infinity.py'),cpu=True)
                saved_receipt('INFINITY_BUILD.json','Build private native BF16 Infinity with measured host-pinned allocator telemetry')
                push('infinity_build_push')
            else:
                run('llama_native_build',command('build_pcie_llama.py'),cpu=True)
                saved_receipt('LLAMA_BUILD.json','Build frozen native balanced3 llama baseline with runtime graphs and reuse disabled')
                run('llama_bf16_conversion',command('prepare_pcie_llama_gguf.py','--out',cell/'conversion'),cpu=True)
                saved_receipt('GGUF_CONVERSION.json','Convert frozen Qwen3 weights locally and verify all144 expert tensors are actual BF16')
                push('llama_build_conversion_push')
            smoke=cell/'smoke'
            run(system+'_smoke',command('run_pcie_baseline_job.py','--system',system,'--smoke','--repeats','5','--timeout','3600','--out',smoke))
            validate_native_smoke(smoke,system);done.append(system+'_smoke_validated')
            primary=cell/'primary'
            run(system+'_full64',command('run_pcie_baseline_job.py','--system',system,'--repeats','5','--timeout','21600','--out',primary))
            run(system+'_validation_archive',command('pcie_baseline_report.py','--system',system,'--root',primary,'--archive','--commit'),cpu=True)
            run(system+'_main_table',command('pcie_baseline_comparison_report.py','--out',EXP/'main_comparison','--commit'),cpu=True)
            push(system+'_result_push')
        table=read_receipt(EXP/'main_comparison/RESULTS.json');assert table['status']=='PASS'
        write(a.out/'result.json',dict(status='PASS',completed=done,systems=['deepspeed','infinity','llama'],main_table=str(EXP/'main_comparison')))
        write(a.out/'status.json',dict(status='PASS',pid=os.getpid(),completed=done))
    except BaseException as exc:
        if current is not None and current.poll() is None:
            if current_cpu:
                os.killpg(current.pid,signal.SIGTERM)
                try:current.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(current.pid,signal.SIGKILL);current.wait()
            else:
                # The owned GPU supervisor observes STOP and performs worker
                # group cleanup, source release and the normal burn transition.
                write(ROOT/'STOP',dict(reason='Baseline continuation interrupted with live GPU child',unix=time.time()))
                try:current.wait(timeout=30)
                except subprocess.TimeoutExpired:pass
        failure=dict(status='FAIL',cause=repr(exc),completed=done,pid=os.getpid(),unix=time.time(),child_pid=current.pid if current else None)
        write(a.out/'failure.json',failure);write(a.out/'status.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predecessor',type=Path,required=True);p.add_argument('--predecessor-pid',type=int,required=True)
    p.add_argument('--predecessor-start-ticks',required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
