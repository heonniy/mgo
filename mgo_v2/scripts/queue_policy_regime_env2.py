"""Historical same-host SHM Env2, only after Env1 attribution process exits."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from finish_policy_regime import identity
import run_timing_stability as h
from run_refactor_measure import publish
BASE=Path('/home/hwlee/mgo-results/policy_regime_20261005')
PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'
ROOT=BASE/'ENV2'
def main(a):
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 ROOT.mkdir(exist_ok=True);(PACKET/'ENV2').mkdir(exist_ok=True)
 with (ROOT/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stamp=identity(a.after_pid);assert stamp and b'finish_policy_regime.py' in Path(f'/proc/{a.after_pid}/cmdline').read_bytes()
  state=dict(status='WAITING_FOR_ENV1_ATTRIBUTION',after_pid=a.after_pid,start_tick=stamp,started_unix=time.time())
  h.write(ROOT/'queue_status.json',state)
  try:
   while identity(a.after_pid)==stamp:
    if (ROOT/'STOP').exists() or (BASE/'STOP').exists():raise RuntimeError('owner STOP')
    if time.time()-state['started_unix']>24*3600:raise TimeoutError('bounded Env1 attribution wait')
    time.sleep(10)
   assert json.loads((BASE/'followup_status.json').read_text())['status']=='ANALYSIS_COMPLETE'
   assert json.loads((PACKET/'POLICY_REGIME_MECHANISMS.json').read_text())['status']=='PASS'
   for setting in ('C30','C60'):
    dst=ROOT/setting;dst.mkdir(exist_ok=True);inputs=dst/'inputs_B128_H64'
    if not inputs.exists():inputs.symlink_to(BASE/setting/'inputs_B128_H64',target_is_directory=True)
    assert inputs.resolve()==(BASE/setting/'inputs_B128_H64').resolve()
   proof=PACKET/'ENV2/POLICY_REGIME_CPU_PROOFS.json';proof.write_bytes((PACKET/'POLICY_REGIME_CPU_PROOFS.json').read_bytes())
   publish('Env2: reuse identical frozen CPU proofs and inputs',[proof])
   env=dict(os.environ,MGO_POLICY_REGIME_ENV='env2')
   for script,output in [('run_policy_regime.py',None),('summarize_policy_regime.py','POLICY_REGIME_TIMING_RESULTS.json'),('summarize_policy_regime_workload.py','POLICY_REGIME_WORKLOAD.json'),('run_policy_regime_profiles.py',None),('summarize_policy_regime_profiles.py','POLICY_REGIME_MECHANISMS.json')]:
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    state.update(status='RUNNING',active=script);h.write(ROOT/'queue_status.json',state)
    with (ROOT/(script+'.log')).open('w') as log:subprocess.run([h.PYTHON,'-u',str(h.P/'scripts'/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if output:publish('Env2: '+output,[PACKET/'ENV2'/output])
   state['status']='ANALYSIS_COMPLETE'
  except BaseException as exc:
   state.update(status='FAIL',error=repr(exc));raise
  finally:
   state['finished_unix']=time.time();h.write(ROOT/'queue_status.json',state)
   receipt=PACKET/'ENV2/EXECUTION.json';h.write(receipt,state);publish('Env2: '+state['status'],[receipt])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True);main(p.parse_args())
