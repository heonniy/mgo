#!/usr/bin/env python3
"""Run three independent single-process CPU replay workers on distinct cores."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path('/home/hwlee/mgo-results/admission_trajectory_controller_breakdown_20261002')

def main():
    jobs={};logs={};finished={}
    affinities={4:180,8:182,16:184}
    assert set(affinities.values()) <= os.sched_getaffinity(0)
    env=dict(os.environ,PYTHONPATH=str(PACKAGE),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while len(finished)<3:
        for batch in (4,8,16):
            if batch not in jobs:
                ready=all((ROOT/'physical'/f'b{batch}_{policy}-rep0-rank{rank}.json').exists()
                          for policy in ('random','hungarian_current') for rank in range(4))
                if ready:
                    log=(ROOT/f'replay-b{batch}.log').open('w');logs[batch]=log
                    command=[sys.executable,str(PACKAGE/'scripts/replay_trajectory.py'),'--batch',str(batch),'--cpu',str(affinities[batch])]
                    jobs[batch]=subprocess.Popen(command,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT)
                    print(json.dumps(dict(started_batch=batch,pid=jobs[batch].pid,cpu=affinities[batch],unix=time.time())),flush=True)
            elif batch not in finished and jobs[batch].poll() is not None:
                finished[batch]=jobs[batch].returncode;logs[batch].close()
                print(json.dumps(dict(finished_batch=batch,exit_code=finished[batch],unix=time.time())),flush=True)
        status=dict(started={str(b):p.pid for b,p in jobs.items()},finished=finished,affinities=affinities,
                    status='FAIL' if any(finished.values()) else 'COMPLETE' if len(finished)==3 else 'RUNNING')
        (ROOT/'replay_status.json').write_text(json.dumps(status,indent=2)+'\n')
        if any(finished.values()): raise SystemExit('A replay failed; inspect its log')
        if len(finished)<3: time.sleep(10)
    print('All 60 single-process replays complete.',flush=True)

if __name__=='__main__':main()
