"""Two cache groups, one arena each; all policies share frozen physical runtime."""
import json,os,time
from pathlib import Path
import run_refactor_measure as m
from batch_comm_common import stop_idle_load,start_idle_load
from policy_regime_paths import ROOT,PACKET,ENVIRONMENT
POLICIES=('BR','OLD_CA','FCA','LA_CA')
def main():
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 m.PACKET=PACKET
 state=dict(environment=ENVIRONMENT,status='RUNNING',completed=[],started_unix=time.time())
 assert json.loads((m.PACKET/'POLICY_REGIME_CPU_PROOFS.json').read_text())['status']=='PASS'
 stop_idle_load()
 try:
  for setting in ('C30','C60'):
   m.ROOT=ROOT/setting;m.h.ROOT=m.ROOT
   state['active']=setting;m.h.write(ROOT/'status.json',state)
   deadline=time.monotonic()+180
   while any(g['temperature_c']>=65 for g in m.h.snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
   cases=[dict(label=p,policy=p,P=2,trigger='T2',horizon=64,overlap=True,partial_precision='bf16',runtime_arm='V3_OPT_PF_OVERLAP',staging_backend='torch',unique_combine=True,async_metadata_inputs=True,isolated_cpu_threads=True,fixed_staging_team=True) for p in POLICIES]
   group=dict(environment=ENVIRONMENT,world=4,gpus=[0,1,4,5],batch=128,horizon=64,paired=True,policy_regime=True,cases=cases)
   result=m.run('POLICY_REGIME_'+setting,group)
   state['completed'].append(dict(setting=setting,unstable=result['result']['unstable']))
  state['status']='TIMING_COMPLETE'
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();m.h.write(ROOT/'status.json',state)
  receipt=m.PACKET/'POLICY_REGIME_EXECUTION.json';m.h.write(receipt,state);m.publish('policy regime: '+state['status'],[receipt])
  if not (ROOT/'STOP').exists():m.h.write(ROOT/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':main()
