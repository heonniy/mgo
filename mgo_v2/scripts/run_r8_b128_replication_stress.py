"""Run the bounded R8/B128/c60 stress search and publish the result."""
import json,os,subprocess,time
from pathlib import Path

P=Path(__file__).resolve().parents[1]
PACKET=P/'experiments/r8_b128_c60_replication_stress_20261004'
ROOT=Path('/home/hwlee/mgo-results/r8_b128_c60_replication_stress_20261004')
POOL=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/pool')
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'
SOURCE=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
BRANCH='codex/r8-b128-c60-replication-stress-20261004'

def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def publish(message):
 subprocess.run(['git','add',str(PACKET)],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  subprocess.run(['git','push','origin','HEAD:'+BRANCH],cwd=P.parent,check=True)

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 state=dict(status='RUNNING',started_unix=time.time(),stage='PREFLIGHT')
 write(PACKET/'status.json',state)
 try:
  for dataset in ('MATH','ShareGPT'):
   receipt=POOL/dataset/'receipt.json'
   if not receipt.exists():raise FileNotFoundError(f'missing completed exact pool: {receipt}')
   obj=json.loads(receipt.read_text())
   if obj.get('status')!='PASS' or obj.get('pool_size',0)<1024:raise RuntimeError(f'invalid pool {dataset}: {obj}')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
   PYTHONPATH=f'{SOURCE}/cpu_deps:{P}:{P}/scripts')
  state.update(stage='CPU_STRESS_SEARCH',pool_source=str(POOL));write(PACKET/'status.json',state)
  log=ROOT/'run.log'
  with log.open('w') as f:
   subprocess.run([PYTHON,'-u',str(P/'scripts/r8_b128_replication_stress_cpu.py')],cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  result=json.loads((PACKET/'RESULT.json').read_text());assert result['status']=='PASS'
  state.update(status='COMPLETE',stage='FINISHED',finished_unix=time.time(),oracle_gate=result['oracle_gate'],winner=result['winner']['manifest'])
  write(PACKET/'status.json',state);publish('results: R8 B128 c60 replication stress upper bound')
 except BaseException as exc:
  state.update(status='FAILED',error=repr(exc),finished_unix=time.time());write(PACKET/'status.json',state);publish('results: preserve R8 B128 stress failure');raise
if __name__=='__main__':main()
