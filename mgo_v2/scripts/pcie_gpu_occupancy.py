"""Maintain our burn while idle; GPU experiments hold explicit process leases."""
import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from pcie_host import ROOT, write

LEASES = ROOT / 'gpu_leases'
BURN_PID = Path('/home/esjung/gpu_burn_0145.pid')
BURN_SCRIPT = '/home/esjung/gpu_burn.py'
CONDA = '/home/esjung/anaconda3/bin/conda'


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        # Fields after the closing comm parenthesis start with proc field 3.
        tail = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if tail[0] == 'Z' or path.stat().st_uid != os.getuid():
            return None
        return tail[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def burn_pid():
    try:
        pid = int(BURN_PID.read_text())
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        if process_identity(pid) and BURN_SCRIPT.encode() in args and os.getpgid(pid) == pid:
            return pid
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    return None


def stop_burn():
    pid = burn_pid()
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 30
        while process_identity(pid):
            if time.monotonic() > deadline:
                raise TimeoutError('Our burn did not exit')
            time.sleep(0.1)


def start_burn():
    pid = burn_pid()
    if pid is not None:
        return pid
    # Do not preempt unrelated work that may have appeared on the targets.
    occupants = subprocess.check_output(['nvidia-smi', '-i', '0,1,4,5',
                  '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if occupants:
        return None
    with Path('/home/esjung/gpu_burn_0145.log').open('a') as log:
        child = subprocess.Popen([CONDA, 'run', '--no-capture-output', '-n', 'base',
                    'python', '-u', BURN_SCRIPT, '--gpus', '0,1,4,5', '--seconds', '0'],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
    BURN_PID.write_text(str(child.pid) + '\n')
    write(ROOT / 'idle_burn.json', dict(pid=child.pid, started=time.time(), physical_gpus=[0,1,4,5]))
    return child.pid


def active_leases():
    LEASES.mkdir(parents=True, exist_ok=True)
    result = []
    for path in LEASES.glob('*.json'):
        try:
            lease = json.loads(path.read_text())
            if process_identity(lease['pid']) == lease['start_ticks']:
                result.append(lease)
            else:
                path.unlink(missing_ok=True)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return result


@contextlib.contextmanager
def gpu_experiment(label):
    LEASES.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    path = LEASES / f'{pid}.json'
    write(path, dict(pid=pid, start_ticks=process_identity(pid), label=label, created=time.time()))
    try:
        stop_burn()
        # The CUDA workers exit asynchronously after the conda parent.
        deadline = time.monotonic() + 30
        while True:
            occupants = subprocess.check_output(['nvidia-smi', '-i', '0,1,4,5',
                        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if not occupants:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f'Target compute occupants remain; refusing to preempt: {occupants}')
            time.sleep(0.2)
        yield
    finally:
        path.unlink(missing_ok=True)
        if not active_leases():
            start_burn()


def supervise():
    import fcntl
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'occupancy.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(ROOT / 'occupancy_manager.json', dict(pid=os.getpid(), started=time.time()))
        while not (ROOT / 'STOP_OCCUPANCY').exists():
            if active_leases():
                stop_burn()
            else:
                start_burn()
            time.sleep(1)


if __name__ == '__main__':
    supervise()
