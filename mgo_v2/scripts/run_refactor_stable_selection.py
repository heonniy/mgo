"""Complete only missing common-stack arms; reuse validated fixed-team V3."""
import json,subprocess,time
from pathlib import Path
import run_refactor_measure as m
from prepare_refactor_arms import ARMS,groups_for
from refactor_fingerprint import capture,assert_equivalent
from batch_comm_common import stop_idle_load,start_idle_load
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/R4_H64')
PACKET=m.PACKET

def main():
 m.ROOT=ROOT;m.h.ROOT=ROOT
 frozen=json.loads((PACKET/'M13_FROZEN_PREFETCH.json').read_text())
 manifest={'world':4,'horizon':64,'gpus':[0,1,4,5],'arms':{},'reused':[]}
 jobs=[];identities={}
 for g in groups_for(frozen,64,[128,256]):
  arm=g['runtime_arm'];b=g['batch'];g.update(world=4,gpus=[0,1,4,5])
  for c in g['cases']:c.update(staging_backend='torch',unique_combine=True,async_metadata_inputs=True,isolated_cpu_threads=True,fixed_staging_team=True)
  stage='R4_H64_FIXED_TEAM' if arm==ARMS[2] else 'R4_FIXED_SEARCH_'+arm
  out=ROOT/f'{stage}_B{b}_H64';manifest['arms'].setdefault(arm,{})[str(b)]=str(out)
  if arm==ARMS[2]:
   state=json.loads((out/'status.json').read_text());assert state['status']=='PASS' and state['group']==g
   assert not state['result']['unstable'];identities[(arm,b)]=state['common_stack'];manifest['reused'].append(str(out))
  else:jobs.append((stage,g))
 for b in (128,256):
  env=m.h.env_for('env1');env.update(CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5')
  identities[('current',b)]=capture(ROOT/f'inputs_B{b}_H64',env)
 assert_equivalent(identities)
 path=PACKET/'STABLE_SEARCH_MANIFEST.json';m.h.write(path,manifest)
 m.publish('plan: bounded stable three-arm selection with V3 reuse',[path])
 state={'status':'RUNNING','completed':[],'started_unix':time.time(),'manifest':str(path)}
 stop_idle_load()
 try:
  for stage,g in jobs:
   state['active']={'stage':stage,'batch':g['batch']};m.h.write(ROOT/'STABLE_SEARCH_STATUS.json',state)
   deadline=time.monotonic()+180
   while any(x['temperature_c']>=65 for x in m.h.snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
   result=m.run(stage,g);state['completed'].append({'stage':stage,'batch':g['batch'],'unstable':result['result']['unstable']})
  state['status']='TIMING_COMPLETE'
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();m.h.write(ROOT/'STABLE_SEARCH_STATUS.json',state)
  receipt=PACKET/'STABLE_SEARCH_EXECUTION.json';m.h.write(receipt,state);m.publish('results: stable-arm search '+state['status'],[receipt])
  if not (ROOT/'STOP').exists():m.h.write(ROOT/'STABLE_SEARCH_resident_models.json',{'processes':start_idle_load(),'unix':time.time()})
if __name__=='__main__':main()
