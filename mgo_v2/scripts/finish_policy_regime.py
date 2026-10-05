"""Wait for the verified timing driver, then audit and capture mechanisms serially."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import run_timing_stability as h
from run_refactor_measure import publish,PACKET
ROOT=Path('/home/hwlee/mgo-results/policy_regime_20261005')
def identity(pid):
 try:
  text=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return None if text[0]=='Z' else text[19]
 except FileNotFoundError:return None
def main(a):
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 with (ROOT/'followup.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stamp=identity(a.timing_pid)
  assert stamp and b'run_policy_regime.py' in Path(f'/proc/{a.timing_pid}/cmdline').read_bytes()
  state=dict(status='WAITING_FOR_TIMING_EXIT',timing_pid=a.timing_pid,timing_start_tick=stamp,started_unix=time.time())
  h.write(ROOT/'followup_status.json',state)
  try:
   while identity(a.timing_pid)==stamp:
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if time.time()-state['started_unix']>12*3600:raise TimeoutError('timing driver still active after bounded wait')
    time.sleep(10)
   assert json.loads((ROOT/'status.json').read_text())['status']=='TIMING_COMPLETE'
   for script,output in [('summarize_policy_regime.py','POLICY_REGIME_TIMING_RESULTS.json'),('run_policy_regime_profiles.py',None),('summarize_policy_regime_profiles.py','POLICY_REGIME_MECHANISMS.json')]:
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    state['status']='RUNNING';state['active']=script;h.write(ROOT/'followup_status.json',state)
    with (ROOT/(script+'.log')).open('w') as log:subprocess.run([h.PYTHON,'-u',str(h.P/'scripts'/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
    if output:publish('policy regime: '+output,[PACKET/output])
   state['status']='ANALYSIS_COMPLETE'
  except BaseException as exc:
   state.update(status='FAIL',error=repr(exc));raise
  finally:
   state['finished_unix']=time.time();h.write(ROOT/'followup_status.json',state)
   receipt=PACKET/'POLICY_REGIME_FOLLOWUP.json';h.write(receipt,state);publish('policy regime followup: '+state['status'],[receipt])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--timing-pid',type=int,required=True);main(p.parse_args())
