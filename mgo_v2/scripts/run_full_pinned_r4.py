"""Guarded R4 full-pinned expert-store E2E A/B.

Uses physical GPUs 0,1,4,5 only.  Stops/restores only this project's idle model
loads on those GPUs and never touches 2,3,6,7.
"""
import os,time,subprocess,signal,json
from pathlib import Path

P=Path(__file__).resolve().parents[1]
EXECUTOR=os.environ.get('MGO_FULL_PINNED_EXECUTOR','H0')
assert EXECUTOR in ('H0','H1b')
LABEL='full_pinned_h0_20261007' if EXECUTOR=='H0' else 'full_pinned_r4_20261006'
PACKET=P/'experiments'/LABEL
ROOT=Path(os.environ.get('MGO_FULL_PINNED_ROOT',f'/home/hwlee/mgo-results/{LABEL}'))
OLD=Path('/home/hwlee/mgo-results/critical_path_admission_followup_20261006/b3/C30')
INPUT_SOURCE=OLD/'inputs_B128_H64'
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'
GPUS=[0,1,4,5]
LOAD=Path('/home/hwlee/mgo-results/model_inference_load_20261003')
IDLE_WORKER=P/'examples/model_inference_load.py'

def write(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)

def host_available():
 return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))

def gpu_state():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=[]
 for line in out.splitlines():
  g,mem,temp,util=[int(float(x.strip())) for x in line.split(',')]
  if g in GPUS:rows.append(dict(gpu=g,used_mib=mem,temp_c=temp,util=util))
 return sorted(rows,key=lambda x:x['gpu'])

def owned_idle(pid):
 try:
  proc=Path(f'/proc/{pid}')
  if proc.stat().st_uid!=os.getuid():return False
  args=proc.joinpath('cmdline').read_bytes().decode().split('\0')
  return any(str(p) in args for p in (
   IDLE_WORKER,
   Path('/home/hwlee/mgo-policy-regime/mgo_v2/examples/model_inference_load.py'),
   Path('/home/hwlee/mgo-deepseek-diagnostics/mgo_v2/examples/model_inference_load.py'),
  ))
 except FileNotFoundError:return False

def stop_target_idle():
 stopped=[]
 if not (LOAD/'processes.json').exists():return stopped
 rows=json.loads((LOAD/'processes.json').read_text())
 for row in rows:
  if row.get('gpu') in GPUS and owned_idle(row['pid']):
   os.kill(row['pid'],signal.SIGTERM);stopped.append(row['gpu'])
 deadline=time.monotonic()+20
 while any(owned_idle(row['pid']) for row in rows if row.get('gpu') in GPUS) and time.monotonic()<deadline:time.sleep(.25)
 for row in rows:
  if row.get('gpu') in GPUS and owned_idle(row['pid']):os.kill(row['pid'],signal.SIGKILL)
 return sorted(set(stopped))

def foreign_on_targets():
 uuid_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
 mapping={u.strip():int(g.strip()) for g,u in (x.split(',') for x in uuid_rows)}
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
 out=[]
 for line in apps:
  u,p=line.split(',');g=mapping[u.strip()];pid=int(p.strip())
  if g in GPUS and not owned_idle(pid):out.append(dict(gpu=g,pid=pid))
 return out

def restore_target_idle(gpus):
 if not gpus:return []
 ps=[]
 envbase=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 for g in gpus:
  # Do not start if any process appeared meanwhile.
  if any(x['gpu']==g for x in foreign_on_targets()):continue
  env=dict(envbase,CUDA_VISIBLE_DEVICES=str(g))
  with (LOAD/f'gpu{g}.log').open('a') as log:
   p=subprocess.Popen([PYTHON,'-u',str(IDLE_WORKER)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  ps.append(dict(gpu=g,pid=p.pid))
 old=json.loads((LOAD/'processes.json').read_text()) if (LOAD/'processes.json').exists() else []
 write(LOAD/'processes.json',[r for r in old if r.get('gpu') not in gpus]+ps)
 return ps

def main():
 assert not ROOT.exists(),'preserve prior attempt; use a new root for rerun'
 assert INPUT_SOURCE.exists(),INPUT_SOURCE
 if host_available()<384*2**30:raise RuntimeError(f'need >=384 GiB host memory headroom, have {host_available()/2**30:.1f} GiB')
 ROOT.mkdir(parents=True);PACKET.mkdir(parents=True,exist_ok=True)
 inputs=ROOT/'inputs_B128_H64';inputs.symlink_to(INPUT_SOURCE)
 state=dict(status='RUNNING',physical_gpus=GPUS,started_unix=time.time(),
  host_available_before=host_available(),initial_gpu_state=gpu_state())
 stopped=stop_target_idle();state['stopped_owned_idle_gpus']=stopped
 foreign=foreign_on_targets()
 if foreign:
  state.update(status='FAIL',error=f'foreign process on target GPU: {foreign}');write(ROOT/'status.json',state);raise RuntimeError(state['error'])
 proc=None
 try:
  deadline=time.monotonic()+180
  while any(x['temp_c']>=65 for x in gpu_state()) and time.monotonic()<deadline:time.sleep(5)
  env=dict(os.environ,PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P/"scripts"}:{P/"examples"}',
   CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',
   OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
   TORCHINDUCTOR_COMPILE_THREADS='2')
  for key in list(env):
   if key.startswith('NCCL_'):del env[key]
  env['NCCL_CUMEM_ENABLE']='0'
  state['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip()
  cmd=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
   str(P/'examples'/('full_pinned_h0_worker.py' if EXECUTOR=='H0' else 'full_pinned_r4_worker.py')),'--inputs',str(inputs),'--output',str(ROOT)]
  state['command']=cmd
  with (ROOT/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state['pid']=proc.pid;write(ROOT/'status.json',state)
   released=set()
   while proc.poll() is None:
    if time.time()-state['started_unix']>1800:raise TimeoutError('full pinned R4 A/B exceeded 30 minutes')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if host_available()<96*2**30:raise RuntimeError('host memory guard: <96 GiB available')
    if (ROOT/'boundary.json').exists():
     row=json.loads((ROOT/'boundary.json').read_text());key=row['key']
     if key not in released and all((ROOT/f'{key}_ready_rank{r}.json').exists() for r in range(4)):
      (ROOT/f'{key}_GO').touch();released.add(key)
    time.sleep(1)
   assert proc.returncode==0,str(ROOT/'run.log')
  result=json.loads((ROOT/'result.json').read_text());assert result['status']=='PASS'
  state.update(status='PASS',result=result)
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc))
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise
 finally:
  state['finished_unix']=time.time();state['host_available_after']=host_available();state['final_gpu_state']=gpu_state()
  restored=restore_target_idle(stopped);state['restored_owned_idle_models']=restored
  write(ROOT/'status.json',state);write(PACKET/'EXECUTION.json',state)
 if state['status']=='PASS':print(json.dumps(state['result'],indent=2))

if __name__=='__main__':main()
