"""Bounded physical comparison of exact decode index construction."""
import json,subprocess,statistics,time,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[1];PACK=P/'experiments/decode_layout_compiled_20261007';ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007');PY='/home/hwlee/sub-moe/phase01/.venv/bin/python';CELL='R4_C30_B32_L512_O64'
def write(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def main():
 state=dict(status='RUNNING',started=time.time());write(PACK/'STATUS.json',state)
 try:
  dirs={}
  for mode in ('reference','compiled'):
   label=f'ours_decode_layout_{mode}_B32_L512_v1';d=ROOT/label;state['current']=label;write(PACK/'STATUS.json',state)
   if not (d/'status.json').exists():
    cmd=[PY,str(P/'scripts/run_headline_job.py'),'--label',label,'--system','Ours','--worker','headline_ours_worker.py','--cell',CELL,'--repeats','2','--prefill-optimized','--prefill-layout-fast','--workloads',str(P/'experiments/main_table_global_workload_20261006/expanded_matrix/WORKLOADS.json')]
    cmd+=['--decode-layout-fast'] if mode=='compiled' else ['--legacy-decode-layout']
    subprocess.run(cmd,check=True)
   assert json.loads((d/'status.json').read_text())['status']=='PASS'
   dirs[mode]=d
  checks=[]
  for rank in range(4):
   v=json.loads((dirs['compiled']/f'decode_layout_validation_rank{rank}.json').read_text());assert v['status']=='PASS' and v['exact_checks']==3024
   for repeat in (1,2):
    a,b=[json.loads((dirs[m]/f'repeat{repeat}_rank{rank}.json').read_text()) for m in ('reference','compiled')]
    row=dict(rank=rank,repeat=repeat,tokens=a['tokens']==b['tokens'],requests=a['request_ids']==b['request_ids'],state=a['validation']['state_hash']==b['validation']['state_hash'],roles=a['validation']['role_hash']==b['validation']['role_hash'],controller=a['validation']['controller']==b['validation']['controller'],h2d_bytes=a['validation']['scheduler']['bytes']==b['validation']['scheduler']['bytes'],h2d_copies=a['validation']['scheduler']['copies']==b['validation']['scheduler']['copies'],no_compile=a['no_compile'] and b['no_compile']);checks.append(row)
  metrics={}
  for mode,d in dirs.items():
   rows=[json.loads((d/f'repeat{i}.json').read_text()) for i in (1,2)]
   metrics[mode]=dict(samples=rows,statistics={k:dict(mean=statistics.mean(x[k] for x in rows),sample_sd=statistics.stdev(x[k] for x in rows),relative_range=(max(x[k] for x in rows)-min(x[k] for x in rows))/statistics.mean(x[k] for x in rows)) for k in ('TTFT','TPOT','E2E')})
   out=PACK/mode;out.mkdir(exist_ok=True)
   for f in d.glob('*.json'):shutil.copy2(f,out/f.name)
  parity=all(all(v for k,v in row.items() if k not in ('rank','repeat')) for row in checks)
  summary=dict(status='PASS' if parity else 'PARITY_FAIL',checks=checks,metrics=metrics,gain_percent={k:(1-metrics['compiled']['statistics'][k]['mean']/metrics['reference']['statistics'][k]['mean'])*100 for k in ('TTFT','TPOT','E2E')},scope='sequential2 repeats each; all samples retained')
  write(PACK/'SUMMARY.json',summary)
  assert parity,'parity failed; inspect SUMMARY.json'
  state.update(status='PASS',finished=time.time(),current=None);write(PACK/'STATUS.json',state)
 except BaseException as e:
  state.update(status='FAILED_NEEDS_REVIEW',error=repr(e),finished=time.time());write(PACK/'STATUS.json',state);raise
if __name__=='__main__':main()
