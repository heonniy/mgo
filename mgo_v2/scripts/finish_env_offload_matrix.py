"""One-shot lifecycle supervisor for an already running physical matrix.

Does not schedule or retry science. Reports partial failure, or validates the
completed matrix, then restores guarded resident models after science exits.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_env_offload_cell import ROOT, PACKET, P, write


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return stat[19]
    except FileNotFoundError:
        return None


def science_pids():
    result = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if str(P/'examples/env_offload_worker.py').encode() in args:
            result.append(int(path.parent.name))
    return result


def checkpoint():
    statuses = []
    for path in ROOT.glob('*_*/status.json'):
        if '_PLAN_' in str(path) or any(s in str(path) for s in ['_COMPILE_', '_MEASURE_', '_COUNTERS_']):
            try:
                row = json.loads(path.read_text())
            except (ValueError, FileNotFoundError):
                continue
            statuses.append(row)
    return dict(
        unix=time.time(),
        passed={phase: sum(r.get('phase') == phase and r['status'] == 'PASS' for r in statuses)
                for phase in ['PLAN', 'COMPILE', 'MEASURE', 'COUNTERS']},
        running=[{k:r.get(k) for k in ['cell','policy','phase','environment','repeat','pid']}
                 for r in statuses if r['status'] == 'RUNNING'],
    )


def publish(message):
    # Refuse to include unrelated staged work in an unattended checkpoint.
    staged = subprocess.check_output(['git','diff','--cached','--name-only'], cwd=P.parent, text=True).splitlines()
    relative = str(PACKET.relative_to(P.parent)) + '/'
    assert all(name.startswith(relative) for name in staged), staged
    subprocess.run(['git','add',str(PACKET)], cwd=P.parent, check=True)
    if subprocess.run(['git','diff','--cached','--quiet'], cwd=P.parent).returncode:
        subprocess.run(['git','commit','-m',message], cwd=P.parent, check=True)
    subprocess.run(['git','push','origin','HEAD:codex/mgo-r4-trajectory-results-20261002'], cwd=P.parent, check=True)


def main(pid):
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'finish_supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert any(a.endswith(b'/run_env_offload_matrix.py') for a in args), args
        original = identity(pid)
        assert original is not None
        while identity(pid) == original:
            state = checkpoint()
            state.update(status='WAITING_FOR_MATRIX', driver_pid=pid, supervisor_pid=os.getpid())
            write(ROOT/'supervisor_status.json', state)
            time.sleep(30)
        # A dead parent is not proof that its GPU children have exited.
        deadline = time.monotonic() + 7200
        while science_pids() and time.monotonic() < deadline:
            time.sleep(15)
        state = checkpoint()
        if science_pids():
            state.update(status='HOLD_SCIENTIFIC_PROCESSES_REMAIN', pids=science_pids())
            write(ROOT/'supervisor_status.json', state)
            return
        completed = 'BOUNDED_MATRIX_COMPLETE' in (ROOT/'matrix_driver.log').read_text()
        state['status'] = 'MATRIX_FAILED_OR_STOPPED'
        if completed:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                       PYTHONPATH='/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:' + str(P/'scripts'))
            with (ROOT/'summary.log').open('w') as log:
                result = subprocess.run([sys.executable,str(P/'scripts/summarize_env_offload.py')],
                                        cwd=P.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['status'] = 'COMPLETE' if result.returncode == 0 else 'REPORT_VALIDATION_FAILED'
            state['summary_exit_code'] = result.returncode
        text = ('# ' + state['status'] + '\n\n' +
                'Accepted phase counts: ' + json.dumps(state['passed']) + '.\n\n' +
                ('See RESULTS.md, validation.json and the separate timing/counter tables.\n' if state['status']=='COMPLETE' else
                 'Partial results only. Inspect matrix_driver.log and summary.log in the raw root; no automatic scientific retry.\n') +
                '\nRaw root: `' + str(ROOT) + '`.\n')
        (PACKET/'STATUS.md').write_text(text)
        matrix = json.loads((PACKET/'matrix.json').read_text())
        matrix['status'] = state['status']
        write(PACKET/'matrix.json', matrix)
        # An explicit stop remains authoritative. Otherwise idle workers retain
        # their own foreign-process, temperature and memory guards.
        if (ROOT/'STOP').exists():
            state['resident_models'] = 'NOT_RESTORED_OWNER_STOP'
        else:
            from batch_comm_common import start_idle_load, LOAD, owned
            processes = start_idle_load()
            deadline = time.monotonic() + 600
            verified = []
            while time.monotonic() < deadline:
                verified = []
                for process in processes:
                    path = LOAD/f"gpu{process['gpu']}.json"
                    if not path.exists():
                        continue
                    row = json.loads(path.read_text())
                    if row.get('pid') == process['pid'] and owned(process['pid']) and row.get('iterations',0)>=2 and time.time()-row.get('unix',0)<90:
                        verified.append({k:row[k] for k in ['gpu','pid','iterations','batch','unix']})
                if len(verified) == len(processes):
                    break
                time.sleep(15)
            state['resident_models'] = dict(requested=processes,verified=verified,
                all_eight_verified=len({r['gpu'] for r in verified})==8)
        write(PACKET/'completion_receipt.json', state)
        write(ROOT/'supervisor_status.json', state)
        try:
            publish('results: finalize bounded physical offload study and GPU handoff')
        except Exception as exc:
            state['publish_error'] = repr(exc)
            write(ROOT/'supervisor_status.json', state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver-pid', type=int, required=True)
    main(parser.parse_args().driver_pid)
