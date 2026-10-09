"""Continue a verified live best64 supervisor into the full seven-policy cohort."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from pcie_host import ROOT, write

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    completed = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    def status(step):
        write(a.out / 'status.json', dict(status='RUNNING', step=step, pid=os.getpid(), completed=completed))
    def stop_check():
        if (ROOT / 'STOP').exists():
            raise RuntimeError('Owner STOP observed')
    def run(label, argv):
        stop_check(); status(label)
        with (a.out / (label + '.log')).open('w') as log:
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(a.out / 'step.json', dict(step=label, pid=child.pid, argv=argv, started_unix=time.time()))
            code = child.wait()
        if code:
            raise RuntimeError(f'{label} exited {code}; inspect preserved attempt before repair')
        completed.append(label); write(a.out / 'completed.json', completed)
    try:
        status('wait_live_best64_supervisor')
        process = psutil.Process(a.predecessor_pid)
        assert 'resume_pcie_maxgain_search.py' in ' '.join(process.cmdline())
        assert str(a.predecessor) in process.cmdline()
        identity = dict(pid=process.pid, create_time=process.create_time(), argv=process.cmdline())
        write(a.out / 'predecessor_identity.json', identity)
        started = time.monotonic()
        while True:
            stop_check()
            state = json.loads((a.predecessor / 'status.json').read_text())
            if state['status'] == 'FAIL':
                raise RuntimeError('Best64 failed; preserve and diagnose before new GPU work')
            try:
                live = psutil.Process(identity['pid'])
                alive = live.create_time() == identity['create_time'] and live.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                assert state['status'] == 'PASS', state
                break
            if time.monotonic() - started > 10860:
                raise TimeoutError('Predecessor wait bound exceeded; no restart performed')
            time.sleep(2)
        for path in (a.predecessor / 'result.json', a.predecessor / 'final/maxgain_validation.json'):
            assert json.loads(path.read_text())['status'] == 'PASS'
        archived = PKG / 'experiments/pcie_topology_ablation_20261009/maxgain64/final/cohort'
        assert json.loads((archived / 'maxgain_validation.json').read_text())['status'] == 'PASS'
        assert (archived / 'warmups/SOURCE_INVENTORY.json').exists()
        completed.append('best64_terminal_validated')
        job = a.out / 'live'
        run('seven_policy_generation', [sys.executable, str(PKG / 'scripts/run_pcie_ours_job.py'),
            '--arm', 'R-NEAR', '--sequence', 'stage2_grouped', '--timeout', '10800', '--out', str(job)])
        run('seven_policy_validation_archive', [sys.executable, str(PKG / 'scripts/pcie_live_cohort_report.py'),
            '--root', str(job), '--archive', '--commit'])
        write(a.out / 'result.json', dict(status='PASS', completed=completed, live_root=str(job),
            scope='Full seven-policy live cohort; external baselines and final consolidation still required'))
        write(a.out / 'status.json', dict(status='PASS', pid=os.getpid(), completed=completed))
    except BaseException as exc:
        receipt = dict(status='FAIL', cause=repr(exc), pid=os.getpid(), completed=completed, unix=time.time())
        write(a.out / 'failure.json', receipt); write(a.out / 'status.json', receipt)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predecessor', type=Path, required=True)
    p.add_argument('--predecessor-pid', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
