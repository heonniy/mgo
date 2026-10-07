"""Serial expanded matrix. Existing receipts are never overwritten."""
import argparse,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];PACK=P/'experiments/main_table_global_workload_20261006/expanded_matrix';ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007');TOOLS=Path('/home/hwlee/mgo-tools/headline-r4');PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'
SYSTEMS=[('ours','Ours',PY,4),('infinity','MoE-Infinity-repaired',str(TOOLS/'infinity-env/bin/python'),1),('deepspeed','DeepSpeed-ZeRO-Inference',str(TOOLS/'base-env/bin/python'),4),('llama_sync','llama.cpp-sync',str(TOOLS/'base-env/bin/python'),1)]
def write(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def jobs():
 cells=json.loads((PACK/'WORKLOADS.json').read_text())['cells'];result=[]
 for spec in cells:
  for worker,system,python,ranks in SYSTEMS:
   result.append(dict(label=f"expanded_{spec['cell']}_{worker}_v1",cell=spec['cell'],worker=f'headline_{worker}_worker.py',system=system,python=python,ranks=ranks))
 return result
def main(a):
 if a.prepare:
  assert not (PACK/'QUEUE.json').exists();write(PACK/'QUEUE.json',dict(jobs=jobs(),status='PREPARED'));return
 state=json.loads((PACK/'QUEUE.json').read_text());state.update(status='RUNNING',started=time.time())
 for job in state['jobs']:
  if (ROOT/'STOP').exists() or (PACK/'STOP').exists():state['status']='STOPPED';break
  status=ROOT/job['label']/'status.json'
  if status.exists():
   old=json.loads(status.read_text());assert old['status'] in ['PASS','FAIL'],'existing active job';job['status']=old['status'];continue
  # The synchronous implementation needs an actual GPU smoke receipt first.
  if job['worker']=='headline_llama_sync_worker.py':
   smoke=ROOT/'expanded_llama_sync_smoke1/status.json'
   if not smoke.exists() or json.loads(smoke.read_text())['status']!='PASS':job['status']='BLOCKED_SMOKE';continue
  state['current']=job['label'];write(PACK/'QUEUE.json',state)
  cmd=[PY,str(P/'scripts/run_headline_job.py'),'--label',job['label'],'--system',job['system'],'--worker',job['worker'],'--cell',job['cell'],'--python',job['python'],'--ranks',str(job['ranks']),'--repeats','5','--timeout','28800','--workloads',str(PACK/'WORKLOADS.json')]
  if job['worker']=='headline_ours_worker.py':cmd+=['--prefill-optimized','--prefill-layout-fast']
  ret=subprocess.run(cmd,check=False);job['returncode']=ret.returncode;job['status']=json.loads(status.read_text())['status'] if status.exists() else 'FAILED_TO_START';write(PACK/'QUEUE.json',state)
  subprocess.run([PY,str(P/'scripts/report_expanded_headline.py')],check=True)
 state.update(status='FINISHED' if state['status']!='STOPPED' else 'STOPPED',finished=time.time(),current=None);write(PACK/'QUEUE.json',state)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');main(p.parse_args())
