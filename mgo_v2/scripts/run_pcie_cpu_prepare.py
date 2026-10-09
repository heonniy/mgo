"""Pause a CPU-only preparation process group during GPU primary samples."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from pcie_host import ROOT, available_bytes, write


def main(a):
    if a.command and a.command[0]=='--':a.command=a.command[1:]
    assert a.command
    a.out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MGO_V2_PHYSICAL_GPUS='',
             CUDA_HOME='/data2/esjung/envs/cuda121',MAX_JOBS='4',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    command=(['bash','-c','kill -STOP $$; exec "$@"','pcie_cpu_prepare',*a.command] if a.start_paused else a.command)
    with (a.out/'worker.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(a.out/'status.json',dict(status='RUNNING',pid=proc.pid,argv=a.command,gpus_visible=''))
        paused=a.start_paused;events=[];started=time.monotonic()
        if paused:
            events.append(dict(action='PAUSE',unix=time.time(),reason='Start paused before importing/scanning'))
            write(a.out/'events.json',events)
        try:
            while proc.poll() is None:
                primary=False
                phase_path=a.gpu_job/'phase.json'
                if phase_path.exists():
                    phase=json.loads(phase_path.read_text());primary=phase.get('phase')=='target' or phase.get('repeat',0)!=0
                # Terminal GPU job permits CPU preparation to resume. Confirm
                # the worker PID is gone as well; a status file alone is weak.
                status_path=a.gpu_job/'status.json'
                if status_path.exists():
                    status=json.loads(status_path.read_text())
                    if status.get('status') in ('PASS','FAIL'):
                        try:os.kill(status['pid'],0)
                        except (ProcessLookupError,KeyError):primary=False
                if primary and not paused:
                    os.killpg(proc.pid,signal.SIGSTOP);paused=True
                    events.append(dict(action='PAUSE',unix=time.time(),reason='GPU target samples'))
                    write(a.out/'events.json',events)
                elif not primary and paused:
                    os.killpg(proc.pid,signal.SIGCONT);paused=False
                    events.append(dict(action='RESUME',unix=time.time(),reason='GPU warmup/diagnostic or worker terminal'))
                    write(a.out/'events.json',events)
                if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP observed')
                if available_bytes()<132*2**30:raise RuntimeError('CPU preparation approached 128-GiB host guard')
                if time.monotonic()-started>a.timeout:raise TimeoutError('CPU preparation wall bound exceeded')
                time.sleep(.2)
            if proc.returncode:raise RuntimeError(f'CPU preparation failed with exit {proc.returncode}')
            write(a.out/'status.json',dict(status='PASS',pid=proc.pid,wall_seconds=time.monotonic()-started,argv=a.command,events=events))
        except BaseException as exc:
            failure=dict(status='FAIL',pid=proc.pid,cause=repr(exc),timestamp=time.time())
            write(a.out/'failure.json',failure);write(a.out/'status.json',failure);raise
        finally:
            try:os.killpg(proc.pid,signal.SIGCONT);os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-job',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=14400);p.add_argument('--start-paused',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER);main(p.parse_args())
