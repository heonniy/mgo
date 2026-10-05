"""Bounded post-timing Nsight run. Never launch alongside primary GPU timing."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import run_timing_stability as h
from batch_comm_common import stop_idle_load,start_idle_load
from prepare_refactor_arms import ARMS,groups_for
from run_refactor_measure import publish
from profile_process_cleanup import track,terminate
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004');FROZEN_PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';PACKET=FROZEN_PACKET/('ENV2' if os.environ.get('MGO_POLICY_REGIME_ENV')=='env2' else '');h.ROOT=ROOT

def main(a):
 global ROOT
 ROOT=a.root;h.ROOT=ROOT;world=len(a.gpus)
 frozen=json.loads((FROZEN_PACKET/'M13_FROZEN_PREFETCH.json').read_text())
 group=next(g for g in groups_for(frozen,a.horizon,[a.batch]) if g['runtime_arm']==a.arm)
 case=(json.loads(a.case_override.read_text()) if getattr(a,'case_override',None) else next(c for c in group['cases'] if c['policy']==a.policy))
 assert case['policy']==a.policy and case['runtime_arm']==a.arm and case['horizon']==a.horizon
 case['staging_backend']=a.staging_backend
 case['unique_combine']=a.unique_combine
 case['profile_cuda_flush_ms']=a.cuda_flush_ms
 case['async_metadata_inputs']=a.async_metadata_inputs
 case['fixed_staging_team']=a.fixed_staging_team
 case['isolated_cpu_threads']=a.fixed_staging_team
 if os.environ.get('MGO_NSYS_BINARY'):
  binary=Path(os.environ['MGO_NSYS_BINARY']).resolve();assert binary.is_file()
  case['profile_nsys_binary']=str(binary)
  case['profile_nsys_version']=subprocess.check_output([str(binary),'--version'],text=True).strip()
  case['profile_disable_device_event_trace']=True
 label=f'{a.stage}_{a.arm}_{a.policy}_B{a.batch}_H{a.horizon}';out=ROOT/label;out.mkdir(exist_ok=False);h.write(out/'case.json',case)
 state=dict(environment=a.environment,world=world,physical_gpus=a.gpus,status='RUNNING',stage=label,case=case,started_unix=time.time(),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip(),primary_timing=False)
 stop_idle_load();proc=None;descendants={}
 def resource_sample(pid=None):
  row=h.sample(pid)
  if case.get('b2_allow_other_gpu_jobs',False):
   uuids={line.split(',')[1].strip():int(line.split(',')[0]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
   apps=[line.split(',') for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()]
   target_pids={int(pid.strip()) for uuid,pid in apps if uuids[uuid.strip()] in a.gpus}
   row['foreign_pids_on_other_gpus']=[p for p in row['foreign_pids'] if p not in target_pids]
   row['foreign_pids']=[p for p in row['foreign_pids'] if p in target_pids]
   row['gpus']=[g for g in row['gpus'] if g['gpu'] in a.gpus]
   state['latest_resource_sample']=row
  return row
 try:
  deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in h.snapshot()['gpus'] if g['gpu'] in a.gpus) and time.monotonic()<deadline:time.sleep(5)
  h.safe(resource_sample(),True);env=h.env_for(a.environment);env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,a.gpus)),MGO_V2_PHYSICAL_GPUS=','.join(map(str,a.gpus)));cache=ROOT/'compile_cache';env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
  cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'scripts/refactor_nsys_rank.py'),'--inputs',str(ROOT/f'inputs_B{a.batch}_H{a.horizon}'),'--case',str(out/'case.json'),'--output',str(out)]
  with (out/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;h.write(out/'status.json',state)
   while proc.poll() is None:
    track(proc.pid,descendants)
    progress_files=[out/f'progress_rank{r}.json' for r in range(world)]
    if all(p.exists() for p in progress_files):
     progress=[json.loads(p.read_text()) for p in progress_files]
     if all(p['stage']=='CAPTURE' for p in progress) and time.time()-max(p['unix'] for p in progress)>180:raise TimeoutError('all ranks made no captured decode progress for 180 seconds')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if time.time()-state['started_unix']>(1800 if a.horizon==8 else 7200):raise TimeoutError('bounded profiling run')
    h.safe(resource_sample(proc.pid));time.sleep(10)
   assert proc.returncode==0,str(out/'run.log')
  rows=[json.loads((out/f'rank{r}.json').read_text()) for r in range(world)];assert all(r['status']=='PASS' for r in rows)
  reports=[out/f'profile_rank{r}.nsys-rep' for r in range(world)];assert all(f.exists() and f.stat().st_size>0 for f in reports)
  state.update(status='PASS',ranks=rows,reports=[str(f) for f in reports],interpretation='capture only; actual kernel/copy interval analysis still required')
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc));raise
 finally:
  if proc and proc.poll() is None:
   track(proc.pid,descendants)
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  terminate(descendants)
  state['finished_unix']=time.time();h.write(out/'status.json',state);receipt=PACKET/(label+'.json');h.write(receipt,state);publish(f'{label}: {state["status"]}',[receipt])
  if not (ROOT/'STOP').exists() and not getattr(a,'defer_idle_restore',False):h.write(out/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--environment',choices=['env1','env2'],default='env1');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)));p.add_argument('--stage',default='M16');p.add_argument('--horizon',type=int,choices=[8,64,256],default=256);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--batch',type=int,choices=[128,256],required=True);p.add_argument('--policy',choices=['BR','LA','OLD_CA','FCA','LA_CA'],default='LA');p.add_argument('--staging-backend',choices=['torch','memmove'],default='torch');p.add_argument('--unique-combine',action='store_true');p.add_argument('--async-metadata-inputs',action='store_true');p.add_argument('--fixed-staging-team',action='store_true');p.add_argument('--case-override',type=Path);p.add_argument('--defer-idle-restore',action='store_true');p.add_argument('--cuda-flush-ms',type=int,choices=[0,600000],default=0);main(p.parse_args())
