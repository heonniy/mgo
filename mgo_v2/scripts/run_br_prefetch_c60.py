"""Bounded three-job queue; one shared model/pinned store per batch."""
import json,subprocess,time,os
from pathlib import Path
from report_br_prefetch_c60 import analyze_batch,report_all
P=Path(__file__).resolve().parents[1];PACK=P/'experiments/br_prefetch_c60_20261007';ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007');PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'
def write(x):
 p=PACK/'STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def main():
 state=dict(status='RUNNING',started=time.time());write(state)
 try:
  for b in (8,16,64):
   labels=json.loads((PACK/'RUN_LABELS.json').read_text()) if (PACK/'RUN_LABELS.json').exists() else {}
   label=labels.get(str(b),f'br_prefetch_C60_B{b}_H256_v1');d=ROOT/label;state.update(batch=b,phase='physical');write(state)
   if not d.exists():subprocess.run([PY,str(P/'scripts/run_headline_job.py'),'--label',label,'--system','Ours','--worker','br_prefetch_ablation_worker.py','--cell',f'R4_C60_B{b}_L256_O257','--workloads',str(PACK/'WORKLOADS.json'),'--timeout','2400'],check=True,env=dict(os.environ,**({'MGO_BR_DIAGNOSTIC_REFERENCE':str(ROOT/'br_prefetch_C60_B8_H256_v1')} if b==8 and label.endswith('_v2') else {})))
   deadline=time.monotonic()+2500
   while True:
    s=json.loads((d/'status.json').read_text())
    if s['status']!='RUNNING':break
    if time.monotonic()>deadline:raise TimeoutError('supervisor status did not finish')
    time.sleep(5)
   assert s['status']=='PASS',s.get('error')
   state.update(phase='analysis');write(state);analyze_batch(b);report_all()
  state.update(status='PASS',phase='complete',finished=time.time());write(state)
 except BaseException as e:
  state.update(status='FAILED',error=repr(e),finished=time.time());write(state);raise
if __name__=='__main__':main()
