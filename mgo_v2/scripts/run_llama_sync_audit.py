"""Bounded llama.cpp thread/resource audit for one frozen cell."""
import argparse,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PACK=P/'experiments/main_table_global_workload_20261006/expanded_matrix'
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
TOOLS=Path('/home/hwlee/mgo-tools/headline-r4')
PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE=str(TOOLS/'base-env/bin/python')
THREADS=(16,32,64)
def main(a):
 if a.build:subprocess.run([PY,str(P/'scripts/build_headline_llama_sync.py')],check=True)
 jobs=[]
 for t in THREADS:
  label=f'llama_thread_audit_{a.cell}_t{t}_v1'
  out=ROOT/label
  if not (out/'status.json').exists():
   cmd=[PY,str(P/'scripts/run_headline_job.py'),'--label',label,'--system','llama.cpp-sync-audit','--worker','headline_llama_sync_worker.py','--cell',a.cell,'--python',BASE,'--ranks','1','--repeats',str(a.repeats),'--timeout','28800','--workloads',str(PACK/'WORKLOADS.json'),'--llama-threads',str(t)]
   subprocess.run(cmd,check=True)
  status=json.loads((out/'status.json').read_text());assert status['status']=='PASS'
  jobs.append((t,out))
 baseline=None;rows=[]
 for t,out in jobs:
  placement=json.loads((out/'placement_audit.json').read_text());assert placement['status']=='PASS'
  samples=[]
  for r in range(1,a.repeats+1):
   x=json.loads((out/f'repeat{r}.json').read_text())
   if baseline is None:baseline=[json.loads((out/f'repeat{k}.json').read_text())['tokens'] for k in range(1,a.repeats+1)]
   assert x['tokens']==baseline[r-1],f'generated token mismatch at threads={t}, repeat={r}'
   samples.append({k:x[k] for k in ('TTFT','TPOT','E2E')})
  rows.append(dict(threads=t,samples=samples,cpu_affinity=json.loads((out/'config.json').read_text())['cpu_affinity'],placement=placement))
 summary=dict(status='PASS',cell=a.cell,repeats=a.repeats,threads=list(THREADS),exact_token_parity_across_threads=True,placement_audit_all=True,metric_recompute_all=True,rows=rows)
 path=ROOT/f'llama_thread_audit_{a.cell}_summary.json';path.write_text(json.dumps(summary,indent=2)+'\n');print(path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',default='R4_C30_B64_L512_O64');p.add_argument('--repeats',type=int,choices=range(1,4),default=2);p.add_argument('--build',action='store_true');main(p.parse_args())
