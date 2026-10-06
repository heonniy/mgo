"""Guarded L256 capture on physical GPUs 0,1,4,5 only."""
import os,time,subprocess,signal,json
from strict_headroom_common import *
import run_timing_stability as h
from strict_headroom_guard import cooldown

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    status=ROOT/'capture_L256_status.json'
    if status.exists():raise RuntimeError('preserve existing capture attempt')
    workers=[];logs=[];state=dict(status='RUNNING',physical_gpus=R4_GPUS,started_unix=time.time(),workers=[])
    try:
        # This runner never launches work on 2,3,6,7.
        snap=cooldown(R4_GPUS)
        state['initial']=snap;write(status,state)
        envbase=dict(os.environ,PYTHONPATH=f'{P}:{P/"scripts"}:{P/"examples"}')
        for gpu in R4_GPUS:
            env=dict(envbase,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=(ROOT/f'capture_L256_gpu{gpu}.log').open('w');logs.append(log)
            proc=subprocess.Popen([h.PYTHON,'-u',str(P/'examples/strict_ttft_capture.py'),'--gpu',str(gpu)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append(proc);state['workers'].append(dict(gpu=gpu,pid=proc.pid))
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('capture worker failed')
            if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
            if time.time()-state['started_unix']>7200:raise TimeoutError('bounded L256 capture')
            time.sleep(5)
        assert all(p.returncode==0 for p in workers);state['status']='PASS'
    except BaseException as exc:
        state.update(status='FAIL',error=repr(exc));raise
    finally:
        for p in workers:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
        for p in workers:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()
        state['finished_unix']=time.time();write(status,state);write(PACKET/'STRICT_L256_CAPTURE.json',state)

if __name__=='__main__':main()
