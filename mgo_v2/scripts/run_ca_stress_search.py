"""Queued exact-capture and CPU stress search; no physical policy timing."""
import os,sys,json,subprocess,signal,time
from pathlib import Path
import psutil
from ca_stress_common import *

def publish(message):
 subprocess.run(['git','add',str(PACKET)],cwd=PACKAGE.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=PACKAGE.parent).returncode:subprocess.run(['git','commit','-m',message],cwd=PACKAGE.parent,check=True)
 push=subprocess.run(['git','push','origin','HEAD:codex/mgo-r4-trajectory-results-20261002'],cwd=PACKAGE.parent)
 if push.returncode:
  subprocess.run(['git','fetch','origin','codex/mgo-r4-trajectory-results-20261002'],cwd=PACKAGE.parent,check=True)
  merged=subprocess.run(['git','merge','--no-edit','FETCH_HEAD'],cwd=PACKAGE.parent)
  if merged.returncode:
   subprocess.run(['git','merge','--abort'],cwd=PACKAGE.parent,check=True);raise RuntimeError('Publish merge conflict; receipts retained')
  subprocess.run(['git','push','origin','HEAD:codex/mgo-r4-trajectory-results-20261002'],cwd=PACKAGE.parent,check=True)

def stage(script):
 if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONPATH=f'{SOURCE}/cpu_deps:{PACKAGE}:{PACKAGE}/scripts')
 with (ROOT/(script+'.log')).open('w') as log:subprocess.run([PYTHON,'-u',str(PACKAGE/'scripts'/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def capture():
 path=ROOT/'capture_status.json'
 if path.exists():assert json.loads(path.read_text())['status']=='PASS','capture requires explicit failure recovery';return
 workers=[];logs=[];state=dict(status='LOADING',started_unix=time.time(),original512_recaptures=0,physical_GPUs=list(range(8)))
 def check(initial=False):
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True)
  rows=[list(map(int,line.split(','))) for line in gpu.splitlines()];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
  assert all(int(pid) in {p.pid for p in workers} for pid in apps.splitlines()),'foreign GPU process'
  available=psutil.virtual_memory().available;assert available>=(512 if initial else 256)*2**30
  assert all(free>(76000 if initial else 8192) and temp<(65 if initial else 85) for _,free,temp in rows)
  if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP')
  return dict(host_available_bytes=available,gpus=rows)
 state['initial']=check(True);write(path,state)
 try:
  for rank in range(8):
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=f'{SOURCE}/cpu_deps:{PACKAGE}:{PACKAGE}/scripts')
   log=(ROOT/f'capture_gpu{rank}.log').open('w');logs.append(log)
   proc=subprocess.Popen([PYTHON,'-u',str(PACKAGE/'examples/ca_stress_capture.py'),'--rank',str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);workers.append(proc)
  state['workers']=[dict(rank=i,pid=p.pid) for i,p in enumerate(workers)]
  while any(p.poll() is None for p in workers):
   for i,p in enumerate(workers):
    if p.poll() not in (None,0):raise RuntimeError(f'Capture GPU{i} exited {p.returncode}')
   state.update(status='CAPTURING_ADDITIONAL_REQUESTS',resources=check(),updated_unix=time.time());write(path,state)
   if time.time()-state['started_unix']>6*3600:raise TimeoutError('six-hour capture limit')
   time.sleep(5)
  assert all(p.returncode==0 for p in workers) and all((ROOT/f'gpu{i}_capture_complete.json').exists() for i in range(8))
  state.update(status='PASS',finished_unix=time.time());write(path,state);write(PACKET/'capture_receipt.json',state)
 except BaseException as exc:state.update(status='FAIL',error=repr(exc),finished_unix=time.time());write(path,state);raise
 finally:
  for p in workers:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  for p in workers:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  for log in logs:log.close()

def main():
 ROOT.mkdir(exist_ok=True);state=dict(status='RUNNING',stage='REQUEST_MANIFESTS',owner_commit='bcff29fc97572201f5b2a1131e514ee20c026322',started_unix=time.time())
 write(PACKET/'status.json',state)
 try:
  stage('check_ca_stress_runtime.py')
  stage('prepare_ca_stress.py');publish('experiment: freeze extended exact request pools preserving original512')
  state['stage']='GPU_CAPTURE';write(PACKET/'status.json',state);capture();publish('results: capture additional exact routes on all eight GPUs')
  state['stage']='TRACE_AUDIT_AND_PACK';write(PACKET/'status.json',state);stage('pack_ca_stress_pool.py');publish('results: validate extended exact trace pools')
  # The GPUs have finished scientific work. Keep the owner's real model load
  # while the remaining search is CPU-only; its independent guards stay active.
  from run_timing_stability import restore
  state['idle_model_handoff_before_CPU_search']=restore()
  state['stage']='CPU_A_AND_B';write(PACKET/'status.json',state);stage('run_ca_stress_cpu.py');publish('results: finish bounded routing prescreen and exact seed replay')
  stage('summarize_ca_stress.py');state.update(status='COMPLETE',stage='FINISHED',finished_unix=time.time());write(PACKET/'status.json',state);publish('results: freeze CA communication stress winners and neutral controls')
 except BaseException as exc:state.update(status='FAILED_OR_STOPPED',error=repr(exc),finished_unix=time.time());write(PACKET/'status.json',state);publish('results: preserve CA stress failure checkpoint');raise
if __name__=='__main__':main()
