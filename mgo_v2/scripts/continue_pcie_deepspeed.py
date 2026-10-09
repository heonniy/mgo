"""Start native ZeRO smoke/full64 only after the seven-policy pipeline terminates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from pcie_host import ROOT,write

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent


def validate_smoke(root):
    assert json.loads((root/'status.json').read_text())['status']=='PASS'
    result=json.loads((root/'result.json').read_text())
    assert result['status']=='PASS' and result['smoke'] and result['system']=='DeepSpeed-ZeRO-Inference'
    for rank in range(4):
        leaf=json.loads((root/f'zero_leaf_rank{rank}.json').read_text())
        assert leaf['status']=='PASS' and leaf['blocks']==48
        calibration=json.loads((root/f'calibration_rank{rank}.json').read_text())
        assert calibration['status']=='PASS' and 0<calibration['all_parameter_peak_bytes']<=17392730112//4
        for repeat in (0,1):
            row=json.loads((root/f'repeat{repeat}_rank{rank}.json').read_text())
            assert row['smoke'] and row['finite_logits'] and row['kv_gpu_resident']
            assert len(row['tokens'])==1 and len(row['tokens'][0])==2
            assert len(row['token_ready_ns'])==2 and row['token_ready_ns'][0]<row['token_ready_ns'][1]
            assert row['parameter_budget_bytes']==17392730112//4
            assert 0<row['all_parameter_peak_bytes']<=row['parameter_budget_bytes']
    write(root/'SMOKE_VALIDATION.json',dict(status='PASS',scope='Four-rank native collective progress, MoE leaf setup, two generated tokens, finite logits, GPU KV and C30 all-parameter bound; not full64 serving performance'))


def main(a):
    a.out.mkdir(parents=True,exist_ok=False);completed=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    def check_stop():
        if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP observed')
    def status(step):write(a.out/'status.json',dict(status='RUNNING',step=step,pid=os.getpid(),completed=completed))
    def run(step,argv):
        check_stop();status(step)
        with (a.out/(step+'.log')).open('w') as log:
            child=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(a.out/'step.json',dict(step=step,pid=child.pid,argv=argv,started_unix=time.time()))
            code=child.wait()
        if code:raise RuntimeError(f'{step} exited {code}; preserve attempt and diagnose before any retry')
        completed.append(step);write(a.out/'completed.json',completed)
    try:
        status('wait_live_seven_policy_pipeline')
        p=psutil.Process(a.predecessor_pid)
        assert 'continue_pcie_stage2.py' in ' '.join(p.cmdline()) and str(a.predecessor) in p.cmdline()
        identity=dict(pid=p.pid,create_time=p.create_time(),argv=p.cmdline());write(a.out/'predecessor_identity.json',identity)
        start=time.monotonic()
        while True:
            check_stop();state=json.loads((a.predecessor/'status.json').read_text())
            if state['status']=='FAIL':raise RuntimeError('Seven-policy predecessor failed; no baseline launch')
            try:
                p=psutil.Process(identity['pid']);alive=p.create_time()==identity['create_time'] and p.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:alive=False
            if not alive:
                assert state['status']=='PASS',state
                break
            if time.monotonic()-start>10920:raise TimeoutError('Predecessor wait bound exceeded; no restart performed')
            time.sleep(2)
        assert json.loads((a.predecessor/'result.json').read_text())['status']=='PASS'
        archived=PKG/'experiments/pcie_topology_ablation_20261009/stage2_grouped/cohort/live_cohort_validation.json'
        receipt=json.loads(archived.read_text());assert receipt['status']=='PASS' and len(receipt['arms'])==7
        completed.append('seven_policy_terminal_validated')
        script=PKG/'scripts/run_pcie_baseline_job.py'
        smoke=a.out/'smoke'
        run('deepspeed_smoke',[sys.executable,str(script),'--system','deepspeed','--smoke','--repeats','5','--timeout','1800','--out',str(smoke)])
        validate_smoke(smoke);completed.append('smoke_validation')
        primary=a.out/'primary'
        run('deepspeed_full64',[sys.executable,str(script),'--system','deepspeed','--repeats','5','--timeout','14400','--out',str(primary)])
        run('deepspeed_validation_archive',[sys.executable,str(PKG/'scripts/pcie_baseline_report.py'),'--system','deepspeed','--root',str(primary),'--archive','--commit'])
        write(a.out/'result.json',dict(status='PASS',completed=completed,primary_root=str(primary),smoke_root=str(smoke),
            scope='Native DeepSpeed baseline; Infinity, llama.cpp and final consolidation still required'))
        write(a.out/'status.json',dict(status='PASS',pid=os.getpid(),completed=completed))
    except BaseException as exc:
        failure=dict(status='FAIL',cause=repr(exc),completed=completed,pid=os.getpid(),unix=time.time())
        write(a.out/'failure.json',failure);write(a.out/'status.json',failure);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predecessor',type=Path,required=True)
    p.add_argument('--predecessor-pid',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
