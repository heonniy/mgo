"""All-policy mechanism captures only after clean primary timing has exited."""
import json,os,subprocess,time
from pathlib import Path
import run_timing_stability as h
from run_refactor_measure import publish,PACKET
from batch_comm_common import start_idle_load
ROOT=Path('/home/hwlee/mgo-results/policy_regime_20261005')
def main():
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 assert json.loads((ROOT/'status.json').read_text())['status']=='TIMING_COMPLETE'
 state=dict(status='RUNNING',completed=[],started_unix=time.time())
 try:
  for setting in ('C30','C60'):
   base=ROOT/setting;cases=json.loads((base/f'POLICY_REGIME_{setting}_B128_H64/cases.json').read_text())
   for case in cases:
    policy=case['policy'];label=f'POLICY_REGIME_PROFILE_{setting}_V3_OPT_PF_OVERLAP_{policy}_B128_H64';out=base/label
    state['active']=label;h.write(ROOT/'profile_status.json',state)
    if (out/'status.json').exists():assert json.loads((out/'status.json').read_text())['status']=='PASS','repair explicitly before retrying failed capture'
    else:
     path=base/f'profile_case_{policy}.json';h.write(path,case)
     cmd=[h.PYTHON,'-u',str(h.P/'scripts/run_refactor_profile.py'),'--root',str(base),'--gpus','0','1','4','5','--stage','POLICY_REGIME_PROFILE_'+setting,'--horizon','64','--arm','V3_OPT_PF_OVERLAP','--batch','128','--policy',policy,'--case-override',str(path),'--staging-backend','torch','--unique-combine','--async-metadata-inputs','--fixed-staging-team','--cuda-flush-ms','600000','--defer-idle-restore']
     with (base/(label+'_driver.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    state['completed'].append(str(out))
  state['status']='CAPTURES_COMPLETE'
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();h.write(ROOT/'profile_status.json',state)
  receipt=PACKET/'POLICY_REGIME_PROFILES.json';h.write(receipt,state);publish('policy regime profiles: '+state['status'],[receipt])
  if not (ROOT/'STOP').exists():h.write(ROOT/'profile_resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':main()
