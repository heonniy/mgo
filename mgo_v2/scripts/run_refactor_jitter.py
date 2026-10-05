"""Bounded diagnostic-only host phase variation capture after primary timing."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import run_timing_stability as h
from batch_comm_common import stop_idle_load,start_idle_load
from prepare_refactor_arms import ARMS,groups_for
from run_refactor_measure import publish
from profile_process_cleanup import track,terminate
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004');PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';h.ROOT=ROOT

def main(a):
 global ROOT
 ROOT=a.root;h.ROOT=ROOT;world=len(a.gpus)
 frozen=json.loads((PACKET/'M13_FROZEN_PREFETCH.json').read_text())
 group=next(g for g in groups_for(frozen,a.horizon,[a.batch]) if g['runtime_arm']==a.arm)
 case=next(c for c in group['cases'] if c['policy']==a.policy)
 case['staging_backend']=a.staging_backend
 case['unique_combine']=a.unique_combine
 case['async_metadata_inputs']=a.async_metadata_inputs
 case['expert_diagnostics']=a.expert_diagnostics
 case['kernel_diagnostics']=a.kernel_diagnostics
 case['gil_switch_abba']=a.gil_switch_abba
 case['thread_affinity_abba']=a.thread_affinity_abba
 assert not (a.gil_switch_abba and a.thread_affinity_abba)
 repeats=(1,2,3,4) if (a.gil_switch_abba or a.thread_affinity_abba) else (1,2)
 assert not (a.expert_diagnostics and a.kernel_diagnostics)
 assert world==4 and a.horizon==64
 label=f'{a.stage}_{a.arm}_{a.policy}_B{a.batch}_H{a.horizon}';out=ROOT/label;out.mkdir(exist_ok=False);h.write(out/'case.json',case)
 state=dict(world=world,physical_gpus=a.gpus,status='RUNNING',stage=label,case=case,started_unix=time.time(),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip(),primary_timing=False,repeats=list(repeats))
 stop_idle_load();proc=None;descendants={}
 try:
  deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in h.snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
  h.safe(h.sample(),True);env=h.env_for('env1');env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,a.gpus)),MGO_V2_PHYSICAL_GPUS=','.join(map(str,a.gpus)));cache=ROOT/'compile_cache';env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
  cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'examples/refactor_jitter_worker.py'),'--inputs',str(ROOT/f'inputs_B{a.batch}_H{a.horizon}'),'--case',str(out/'case.json'),'--output',str(out)]
  with (out/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;h.write(out/'status.json',state)
   while proc.poll() is None:
    track(proc.pid,descendants)
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if time.time()-state['started_unix']>(1800 if a.horizon==8 else 7200):raise TimeoutError('bounded profiling run')
    h.safe(h.sample(proc.pid));time.sleep(10)
   assert proc.returncode==0,str(out/'run.log')
  rows=[json.loads((out/f'diagnostic_r{rep}_rank{r}.json').read_text()) for rep in repeats for r in range(world)]
  assert all(row['status']=='PASS' and row['primary_timing'] is False for row in rows)
  for rank in range(world):
   first,second=rows[rank],rows[world+rank]
   assert all(rows[(rep-1)*world+rank]['controller_counters']==first['controller_counters'] for rep in repeats)
   # Timing-dependent priority labels are observations, not invalid data.
   state.setdefault('scheduler_comparison',[]).append(dict(rank=rank,identical=first['scheduler_metrics']==second['scheduler_metrics'],first=first['scheduler_metrics'],second=second['scheduler_metrics']))
   assert first['controller_counters']==second['controller_counters']
  state.update(status='PASS',records=[str(out/f'diagnostic_r{rep}_rank{r}.json') for rep in repeats for r in range(world)],interpretation='Bounded diagnostic same-policy generations; no primary gain claim. Host phase attribution still requires analysis.')
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
  if not (ROOT/'STOP').exists():h.write(out/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)));p.add_argument('--stage',default='R4_HOST_VARIATION');p.add_argument('--horizon',type=int,choices=[8,64,256],default=256);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--batch',type=int,choices=[128,256],required=True);p.add_argument('--policy',choices=['BR','LA'],default='LA');p.add_argument('--staging-backend',choices=['torch','memmove'],default='torch');p.add_argument('--unique-combine',action='store_true');p.add_argument('--async-metadata-inputs',action='store_true');p.add_argument('--expert-diagnostics',action='store_true');p.add_argument('--kernel-diagnostics',action='store_true');p.add_argument('--gil-switch-abba',action='store_true');p.add_argument('--thread-affinity-abba',action='store_true');main(p.parse_args())
