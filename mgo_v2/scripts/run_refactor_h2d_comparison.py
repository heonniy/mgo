"""Post-timing LA-only three-arm H2D comparison; separate from primary timing."""
import json,subprocess,time
from pathlib import Path
from prepare_refactor_arms import ARMS,PACKET
from batch_comm_common import start_idle_load
from run_refactor_measure import publish
import run_timing_stability as h
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/R4_H64')

def main():
 assert json.loads((ROOT/'STABLE_SEARCH_STATUS.json').read_text())['status']=='TIMING_COMPLETE'
 timing=json.loads((PACKET/'STABLE_THREE_ARM_RESULTS.json').read_text());assert timing['common_stack_equivalence']=='PASS'
 jobs=[(a,b,'LA') for a in ARMS for b in (128,256)]
 if timing['winner']:jobs.extend((timing['winner'],b,'BR') for b in (128,256))
 state=dict(status='RUNNING',jobs=[dict(arm=a,batch=b,policy=p) for a,b,p in jobs],completed=[],started_unix=time.time())
 try:
  for arm,batch,policy in jobs:
   label=f'R4_FIXED_H2D_{arm}_{policy}_B{batch}_H64';out=ROOT/label
   state['active']=label;h.write(ROOT/'H2D_COMPARISON_STATUS.json',state)
   if (out/'status.json').exists():assert json.loads((out/'status.json').read_text())['status']=='PASS','failed captures require an explicitly repaired new stage'
   else:
    cmd=[h.PYTHON,'-u',str(h.P/'scripts/run_refactor_profile.py'),'--root',str(ROOT),'--gpus','0','1','4','5','--stage','R4_FIXED_H2D','--horizon','64','--arm',arm,'--batch',str(batch),'--policy',policy,'--staging-backend','torch','--unique-combine','--async-metadata-inputs','--fixed-staging-team','--cuda-flush-ms','600000','--defer-idle-restore']
    with (ROOT/(label+'_driver.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
   state['completed'].append(str(out))
  state['status']='CAPTURES_COMPLETE'
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();h.write(ROOT/'H2D_COMPARISON_STATUS.json',state)
  receipt=PACKET/'H2D_COMPARISON_EXECUTION.json';h.write(receipt,state);publish('profile: fixed-team H2D comparison '+state['status'],[receipt])
  if not (ROOT/'STOP').exists():h.write(ROOT/'H2D_COMPARISON_resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':main()
