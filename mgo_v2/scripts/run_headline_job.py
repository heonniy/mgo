"""Exclusive GPU job supervisor with memory guards and resource receipts."""
import argparse,os,time,subprocess,signal,json
from pathlib import Path
import run_full_pinned_r4 as c
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
def main(a):
 requested_timeout=a.timeout
 # The small native llama warmup takes483s; the large cell has8x input
 # tokens. Allow its unchanged four batches to finish under a finite bound.
 if a.worker=='headline_llama_worker.py' and a.cell=='R4_C30_B64_L512_O64' and not a.smoke:
  a.timeout=max(a.timeout,14400)
 out=ROOT/a.label;assert not out.exists(),'new label required to preserve attempts';assert c.host_available()>=384*2**30
 out.mkdir(parents=True);state=dict(status='RUNNING',started=time.time(),system=a.system,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.P.parent,text=True).strip());stopped=[];proc=None
 state.update(requested_timeout_seconds=requested_timeout,effective_timeout_seconds=a.timeout)
 try:
  stopped=c.stop_target_idle();assert not c.foreign_on_targets()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2',PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{c.P}:{c.P/"scripts"}:{c.P/"examples"}')
  for k in list(env):
   if k.startswith('NCCL_'):del env[k]
  env['NCCL_CUMEM_ENABLE']='0'
  env['PYTHONFAULTHANDLER']='1'
  command=[a.python,'-u']
  if a.ranks>1:command+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.ranks}']
  command +=[str(c.P/'examples'/a.worker),'--cell',a.cell,'--output',str(out)]+(['--smoke'] if a.smoke else [])
  state['command']=command
  with (out/'run.log').open('w') as log,(out/'resources.jsonl').open('w') as resources:
   proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;c.write(out/'status.json',state)
   while proc.poll() is None:
    if time.time()-state['started']>a.timeout:raise TimeoutError('configured job bound')
    if (ROOT/'STOP').exists() or (out/'STOP').exists():raise RuntimeError('owner STOP')
    if c.host_available()<96*2**30:raise RuntimeError('host available below96 GiB')
    phase=json.loads((out/'phase.json').read_text()) if (out/'phase.json').exists() else {'phase':'loading'}
    resources.write(json.dumps(dict(unix=time.time(),phase=phase,gpus=c.gpu_state(),host_available=c.host_available()))+'\n');resources.flush();time.sleep(1)
   state['worker_returncode']=proc.returncode
   assert proc.returncode==0,f'worker exited {proc.returncode}; see run.log'
  result=json.loads((out/'result.json').read_text());assert result['status']=='PASS';state.update(status='PASS',result=result)
 except BaseException as e:
  state.update(status='FAIL',error=repr(e))
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise
 finally:state['finished']=time.time();state['restored']=c.restore_target_idle(stopped);c.write(out/'status.json',state)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--system',required=True);p.add_argument('--worker',required=True);p.add_argument('--cell',required=True);p.add_argument('--python',default=c.PYTHON);p.add_argument('--ranks',type=int,default=4);p.add_argument('--timeout',type=int,default=3600);p.add_argument('--smoke',action='store_true');main(p.parse_args())
