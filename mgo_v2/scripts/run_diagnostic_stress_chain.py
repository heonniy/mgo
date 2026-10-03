"""Owner-authorized diagnostic -> CA stress search -> resident model handoff."""
import json,os,subprocess,time
from pathlib import Path
from run_timing_stability import ROOT,PACKET,P,PYTHON,write,snapshot,restore
FOLLOWUP=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004')
FOLLOWUP.mkdir(exist_ok=True)

def main():
 state=dict(status='RESUMING_DIAGNOSTIC',followup_commit='bcff29fc97572201f5b2a1131e514ee20c026322',automatic_policy_timing=False)
 write(FOLLOWUP/'chain_status.json',state)
 deadline=time.monotonic()+180
 while any(g['temperature_c']>=65 for g in snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
 with (ROOT/'amendment_resume.log').open('w') as log:
  code=subprocess.call([PYTHON,'-u',str(P/'scripts/run_timing_stability_amended.py')],stdout=log,stderr=subprocess.STDOUT)
 diagnostic=json.loads((PACKET/'status.json').read_text());state.update(diagnostic_exit_code=code,diagnostic_status=diagnostic['status'])
 if not diagnostic['status'].startswith('HARNESS_') or (ROOT/'STOP').exists():
  state.update(status='DIAGNOSTIC_REQUIRES_ATTENTION',resident_models=restore());write(FOLLOWUP/'chain_status.json',state);return
 state['status']='WAITING_FOR_PREPARED_STRESS_RUNNER';write(FOLLOWUP/'chain_status.json',state)
 deadline=time.monotonic()+1800
 while not (FOLLOWUP/'READY').exists() and time.monotonic()<deadline:
  if (FOLLOWUP/'STOP').exists():break
  time.sleep(10)
 if not (FOLLOWUP/'READY').exists() or (FOLLOWUP/'STOP').exists():
  state.update(status='FOLLOWUP_NOT_READY_OR_STOPPED',resident_models=restore());write(FOLLOWUP/'chain_status.json',state);return
 state['status']='RUNNING_CA_STRESS_SEARCH';write(FOLLOWUP/'chain_status.json',state)
 try:
  with (FOLLOWUP/'driver.log').open('w') as log:
   code=subprocess.call([PYTHON,'-u',str(P/'scripts/run_ca_stress_search.py')],stdout=log,stderr=subprocess.STDOUT)
  state.update(status='COMPLETE' if code==0 else 'FOLLOWUP_FAILED',followup_exit_code=code)
 finally:
  state['resident_models']=restore();write(FOLLOWUP/'chain_status.json',state)
if __name__=='__main__':main()
