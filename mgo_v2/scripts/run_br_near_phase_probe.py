"""Bounded, exclusive one-cell paired diagnostic; restores owned model load."""
import os,json,signal,subprocess,time
from pathlib import Path
import run_full_pinned_r4 as common
P=common.P;ROOT=Path('/home/hwlee/mgo-results/br_near_phase_probe_20261007');PACKET=P/'experiments/r4_br_near_h0_20261007/diagnostic'
def main():
 assert not ROOT.exists(),'preserve previous attempts'
 assert common.host_available()>=384*2**30
 ROOT.mkdir(parents=True)
 state=dict(status='RUNNING',started_unix=time.time(),physical_gpus=[0,1,4,5],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip())
 stopped=[];proc=None
 try:
  stopped=common.stop_target_idle();assert not common.foreign_on_targets()
  deadline=time.monotonic()+180
  while any(x['temp_c']>=65 for x in common.gpu_state()) and time.monotonic()<deadline:time.sleep(5)
  env=dict(os.environ,PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P/"scripts"}:{P/"examples"}',CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2')
  for k in list(env):
   if k.startswith('NCCL_'):del env[k]
  env['NCCL_CUMEM_ENABLE']='0'
  cmd=[common.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(P/'examples/br_near_phase_worker.py'),'--output',str(ROOT)];state['command']=cmd
  with (ROOT/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;common.write(ROOT/'status.json',state)
   while proc.poll() is None:
    if time.time()-state['started_unix']>1800:raise TimeoutError('30-minute bound')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if common.host_available()<96*2**30:raise RuntimeError('host memory guard')
    time.sleep(2)
   assert proc.returncode==0,'see run.log'
  state.update(status='PASS',result=json.loads((ROOT/'result.json').read_text()))
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc))
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise
 finally:
  state['finished_unix']=time.time();state['restored_owned_idle_models']=common.restore_target_idle(stopped)
  common.write(ROOT/'status.json',state);common.write(PACKET/'EXECUTION.json',state)
if __name__=='__main__':main()
