import os,subprocess,json
from pathlib import Path
import run_full_pinned_r4 as c
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
if __name__=='__main__':
 stopped=[];state={'status':'RUNNING'}
 try:
  stopped=c.stop_target_idle();assert not c.foreign_on_targets()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{c.P}:{c.P/"scripts"}',OMP_NUM_THREADS='1',NCCL_CUMEM_ENABLE='0')
  for name in ['test_live_metadata.py','test_async_metadata.py']:
   with (ROOT/(name+'.log')).open('w') as f:subprocess.run([c.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(c.P/'scripts'/name)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180,check=True)
  state['status']='PASS'
 except Exception as e:state.update(status='FAIL',error=repr(e));raise
 finally:state['restored']=c.restore_target_idle(stopped);c.write(ROOT/'metadata_preflight.json',state)
