"""Run only the owner-selected c60/s0 packet."""
import json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[1]
ROOT=Path('/home/hwlee/mgo-results/transport_stack_remeasure_20261004')
PACKET=P/'experiments/transport_stack_remeasure_20261004'
def write(x):(PACKET/'queue.json').write_text(json.dumps(x,indent=2)+'\n')
def main():
 state=dict(status='RUNNING',owner_followup_commit='70780ff219594355c09843b9b15a6d38764edf05',order=['c60_s0'],active=None)
 for arm,script in [('c60_s0','run_transport_stack_packet.py')]:
  state['active']=arm;write(state)
  with (ROOT/(arm+'_driver.log')).open('w') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(P/'scripts'/script)],stdout=log,stderr=subprocess.STDOUT)
   state['pid']=proc.pid;write(state);rc=proc.wait()
  result=json.loads((PACKET/arm/'status.json').read_text());state[arm]=dict(exit_code=rc,status=result['status'])
  if rc or result['status'] not in ('COMPLETE','UNSTABLE_STOP'):
   state['status']='STOPPED_ON_FAILURE';write(state);return
 state.update(status='COMPLETE',active=None,finished_unix=time.time());write(state)
if __name__=='__main__':main()
