"""Owner-authorized capture -> CPU screen -> preparation -> strict physical queue."""
import os,time,subprocess,json
from strict_headroom_common import *
from batch_comm_common import stop_idle_load,start_idle_load
PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'

def publish(message,files):
 files=[str(f) for f in files if f.exists()]
 if not files:return
 subprocess.run(['git','add',*files],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  if subprocess.run(['git','push','origin','HEAD:refs/heads/codex/policy-regime-20261005'],cwd=P.parent).returncode:
   write(ROOT/'publication_pending.json',dict(message=message))

def main():
 state=dict(status='RUNNING',stage='capture_L256',started_unix=time.time(),owner_authorized_R8_GPUs=list(range(8)))
 env=dict(os.environ,MGO_STRICT_R8_AUTHORIZED='1',MGO_STRICT_R8_GPUS='0,1,2,3,4,5,6,7')
 env['PYTHONPATH']=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P}/scripts:{P}/examples'
 def save():write(ROOT/'pipeline_status.json',state)
 save()
 try:
  deadline=time.monotonic()+7200
  while True:
   if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
   cap=json.loads((ROOT/'capture_L256_status.json').read_text())
   if cap['status']=='PASS':break
   assert cap['status']=='RUNNING',cap
   if time.monotonic()>deadline:raise TimeoutError('capture wait')
   time.sleep(5)
  publish('strict headroom: L256 capture complete',[PACKET/'STRICT_L256_CAPTURE.json'])
  write(ROOT/'resident_models.json',dict(processes=start_idle_load()))
  for script,stage in [('strict_headroom_screen.py','S0'),('prepare_strict_headroom_cases.py','CPU_inputs'),('run_strict_headroom_physical.py','S1_S2')]:
   state['stage']=stage;save()
   if stage=='S1_S2':stop_idle_load()
   with (ROOT/(stage+'.log')).open('w') as log:
    subprocess.run([PY,'-u',str(P/'scripts'/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   publish('strict headroom: '+stage+' complete',list(PACKET.glob('STRICT_*.json')))
  import report_strict_headroom
  report_strict_headroom.main()
  publish('results: strict headroom TTFT analysis',[PACKET/'ANALYSIS.json',PACKET/'ANALYSIS.md'])
  state['status']='PASS'
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();save();write(PACKET/'EXECUTION_STATUS.json',state)
  write(ROOT/'resident_models.json',dict(processes=start_idle_load()))
  publish('strict headroom: pipeline '+state['status'],[PACKET/'EXECUTION_STATUS.json'])
if __name__=='__main__':main()
