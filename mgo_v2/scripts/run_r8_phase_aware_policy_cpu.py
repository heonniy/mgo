"""Guarded CPU-only driver for R8 B128/B256 phase-aware policy replay."""
import json,os,subprocess,time
from pathlib import Path
import psutil

P=Path(__file__).resolve().parents[1]
PACKET=P/'experiments/r8_real_replica_batch_scaling_20261004'
ROOT=Path('/home/hwlee/mgo-results/r8_real_replica_batch_scaling_20261004')
POOL=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/pool')
SOURCE=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'
BRANCH='codex/r8-b128-b256-real-replica-20261004'

def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def publish(message):
 subprocess.run(['git','add',str(PACKET)],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  result=subprocess.run(['git','push','origin','HEAD:'+BRANCH],cwd=P.parent)
  if result.returncode:write(ROOT/'publication_pending.json',dict(status='LOCAL_COMMIT_SAVED_PUSH_PENDING',unix=time.time()))

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 state=dict(status='RUNNING',stage='PREFLIGHT',started_unix=time.time(),cpu_only=True)
 write(PACKET/'status.json',state)
 try:
  for dataset in ('MATH','ShareGPT'):
   receipt=POOL/dataset/'receipt.json'
   if not receipt.exists():raise FileNotFoundError(receipt)
   obj=json.loads(receipt.read_text())
   if obj.get('status')!='PASS' or obj.get('pool_size')!=2048:
    raise RuntimeError(f'B256 requires exact 2048 pool for {dataset}: {obj}')
  if psutil.virtual_memory().available<256*2**30:
   raise RuntimeError('host available memory below 256 GiB guard')
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
   state['note']='GPU processes present but CPU-only experiment does not use CUDA'
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
           PYTHONPATH=f'{SOURCE}/cpu_deps:{P}:{P}/scripts')
  state['stage']='CPU_REPLAY';write(PACKET/'status.json',state)
  with (ROOT/'run.log').open('w') as log:
   proc=subprocess.Popen([PYTHON,'-u',str(P/'scripts/r8_phase_aware_policy_cpu.py')],
                         cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
   state['pid']=proc.pid
   while proc.poll() is None:
    avail=psutil.virtual_memory().available
    rss=psutil.Process(proc.pid).memory_info().rss if psutil.pid_exists(proc.pid) else 0
    state.update(updated_unix=time.time(),host_available_bytes=avail,driver_rss_bytes=rss)
    write(PACKET/'status.json',state)
    if avail<128*2**30:
     proc.terminate()
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
     raise RuntimeError('host memory guard <128 GiB during replay')
    time.sleep(5)
  if proc.returncode:raise RuntimeError(f'CPU replay failed; inspect {ROOT/"run.log"}')
  result=json.loads((PACKET/'RESULT.json').read_text())
  if result.get('status')!='PASS':raise RuntimeError('result status != PASS')
  state.update(status='COMPLETE',stage='FINISHED',finished_unix=time.time(),
               implementation_validation=json.loads((PACKET/'implementation_validation.json').read_text())['status'])
  write(PACKET/'status.json',state);publish('results: phase-aware R8 B128 B256 policy replay')
 except BaseException as exc:
  state.update(status='FAILED',error=repr(exc),finished_unix=time.time())
  write(PACKET/'status.json',state);publish('results: preserve phase-aware policy replay failure');raise

if __name__=='__main__':main()
