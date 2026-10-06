"""Exclusive R4 driver for eight requested policy cells, bounded to one hour."""
import os,json,signal,subprocess,time,shutil
from pathlib import Path
import run_full_pinned_r4 as common
P=common.P;ROOT=Path('/home/hwlee/mgo-results/r4_br_near_h0_20261007');PACKET=P/'experiments/r4_br_near_h0_20261007'
def write(p,row):common.write(p,row)
def publish(paths,message):
 subprocess.run(['git','add',*[str(p) for p in paths]],cwd=P.parent,check=True)
 subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
 subprocess.run(['git','push','origin','HEAD:codex/main-table-global-workload-20261006'],cwd=P.parent,check=True)
def main():
 assert (ROOT/'manifest.json').exists() and not (ROOT/'status.json').exists()
 assert common.host_available()>=384*2**30
 state=dict(status='RUNNING',started_unix=time.time(),physical_gpus=[0,1,4,5],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip(),completed_cells=[])
 stopped=[];proc=None
 try:
  stopped=common.stop_target_idle();assert not common.foreign_on_targets()
  deadline=time.monotonic()+180
  while any(x['temp_c']>=65 for x in common.gpu_state()) and time.monotonic()<deadline:time.sleep(5)
  env=dict(os.environ,PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P/"scripts"}:{P/"examples"}',CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2')
  for k in list(env):
   if k.startswith('NCCL_'):del env[k]
  env['NCCL_CUMEM_ENABLE']='0'
  cmd=[common.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(P/'examples/r4_br_near_h0_worker.py'),'--output',str(ROOT)]
  # B16/B64 frozen-gate metadata parity is checked before loading the model.
  with (ROOT/'metadata_preflight.log').open('w') as testlog:
   subprocess.run([common.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(P/'scripts/test_async_metadata.py')],env=env,stdout=testlog,stderr=subprocess.STDOUT,timeout=120,check=True)
  state['metadata_preflight']='PASS'
  state['command']=cmd
  with (ROOT/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(ROOT/'status.json',state);released=set()
   while proc.poll() is None:
    if time.time()-state['started_unix']>3600:raise TimeoutError('one-hour matrix bound')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if common.host_available()<96*2**30:raise RuntimeError('host memory guard')
    if (ROOT/'boundary.json').exists():
     b=json.loads((ROOT/'boundary.json').read_text());key=b['key']
     if key not in released and all((ROOT/f'{key}_ready_rank{r}.json').exists() for r in range(4)):(ROOT/f'{key}_GO').touch();released.add(key)
    for label in ['B16_L256','B16_L512','B64_L256','B64_L512']:
     f=ROOT/label/'result.json'
     if f.exists() and label not in state['completed_cells']:
      dest=PACKET/f'{label}_RESULT.json';shutil.copy2(f,dest)
      try:publish([dest],f'results: H0 full pinned BR Near {label} single-shot PASS')
      except Exception as exc:state.setdefault('publish_errors',[]).append(repr(exc))
      state['completed_cells'].append(label);write(ROOT/'status.json',state)
    time.sleep(1)
   assert proc.returncode==0,'see run.log'
  result=json.loads((ROOT/'result.json').read_text());assert result['status']=='PASS';state.update(status='PASS',result=result)
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc))
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise
 finally:
  state['finished_unix']=time.time();state['restored_owned_idle_models']=common.restore_target_idle(stopped)
  write(ROOT/'status.json',state);write(PACKET/'EXECUTION.json',state)
if __name__=='__main__':main()
