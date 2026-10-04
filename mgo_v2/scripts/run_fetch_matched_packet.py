"""CPU search then the separately prepared bounded fixed-route GPU screen."""
from fetch_matched_common import *
import subprocess,time
from run_ca_stress_search import publish as old_publish
import run_ca_stress_search as publishing
publishing.PACKET=PACKET

def main():
 ROOT.mkdir(exist_ok=True);state=dict(status='RUNNING',stage='CPU_SEARCH',owner_commit='81dfb0c9a6dd7b41dbdd1a4b4b46ec6d2998cd42',started_unix=time.time(),new_trace_capture=False,horizon=64);write(PACKET/'status.json',state)
 try:
  env=dict(os.environ,PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P}/scripts')
  with (ROOT/'CPU_search.log').open('w') as f:subprocess.run([PYTHON,'-u',str(P/'scripts/fetch_matched_search.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  old_publish('results: decode64 fetch-matched peer and critical search')
  winners=json.loads((PACKET/'winners.json').read_text())
  if winners:
   state['stage']='WAITING_FOR_PHYSICAL_RUNNER';write(PACKET/'status.json',state)
   deadline=time.monotonic()+3600
   while not (ROOT/'PHYSICAL_READY').exists():
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if time.monotonic()>deadline:raise TimeoutError('physical runner preparation deadline')
    time.sleep(10)
   state['stage']='PHYSICAL_SCREEN';write(PACKET/'status.json',state)
   with (ROOT/'physical_driver.log').open('w') as f:subprocess.run([PYTHON,'-u',str(P/'scripts/run_fetch_matched_physical.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  else:
   (PACKET/'RESULTS.md').write_text('# No exact fetch-matched candidates\n\nThe bounded decode64 search found no eligible BR/CA pair. No physical run or expanded search was launched. See fetch_match_validation.json.\n')
  state.update(status='COMPLETE',stage='FINISHED',finished_unix=time.time())
 except BaseException as exc:state.update(status='FAILED_OR_STOPPED',error=repr(exc),finished_unix=time.time());raise
 finally:
  from run_timing_stability import restore
  state['resident_models']=restore();write(PACKET/'status.json',state);old_publish('results: decode64 pivot checkpoint and GPU handoff')
if __name__=='__main__':main()
