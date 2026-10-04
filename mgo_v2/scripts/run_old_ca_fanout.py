"""Three frozen workloads, Env2 only, R4 then R8, no full per-run warmup."""
from prepare_old_ca_fanout import *
import subprocess,signal,time
import run_timing_stability as h
import run_ca_stress_search as publication
h.ROOT=ROOT;publication.PACKET=PACKET

def publish(message):publication.publish(message)
def env_for(world,debug=False,out=None):
 env=h.env_for('env2');env['CUDA_VISIBLE_DEVICES']='0,1,4,6' if world==4 else '0,1,2,3,4,5,6,7';cache=ROOT/'compile_cache'/f'R{world}';cache.mkdir(parents=True,exist_ok=True)
 env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P}/scripts:{P}/examples')
 if debug:env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(out/'nccl-%h-%p.log'))
 return env

def run(world,preflight=False):
 label=f'R{world}_'+('preflight' if preflight else 'physical');out=ROOT/label
 if (out/'status.json').exists():assert json.loads((out/'status.json').read_text())['status']=='PASS';return out
 out.mkdir(exist_ok=False);h.safe(h.sample(),True);worker='env_offload_preflight.py' if preflight else 'old_ca_fanout_worker.py'
 command=[PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'examples'/worker),'--output',str(out)]
 if not preflight:command+=['--plans',str(ROOT/'physical_plans.json')]
 state=dict(status='RUNNING',world=world,physical_gpus=[0,1,4,6] if world==4 else list(range(8)),command=command,started_unix=time.time(),timed_boundaries=[])
 waiting=None
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(command,env=env_for(world,preflight,out),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(out/'status.json',state)
  try:
   last_scan=0
   while proc.poll() is None:
    if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP')
    if time.time()-state['started_unix']>4*3600:raise TimeoutError('four-hour per-rank-count bound')
    if waiting:
     if all((out/f'{waiting}_rank{r}.json').exists() for r in range(world)):waiting=None
     else:
      if time.time()-state['timed_boundaries'][-1]['GO_unix']>1800:raise TimeoutError('30-minute timed-run bound')
      time.sleep(.5);continue
    if not preflight and (out/'phase.json').exists():
     phase=json.loads((out/'phase.json').read_text());state['worker_phase']=phase
     if phase['status']=='READY' and not (out/('GO_'+phase['label'])).exists():
      before=h.sample(proc.pid,True);h.safe(before);waiting=phase['label'];state['timed_boundaries'].append(dict(label=waiting,GO_unix=time.time(),before=before));write(out/'status.json',state);(out/('GO_'+waiting)).touch();continue
    if time.monotonic()-last_scan>=10:h.safe(h.sample(proc.pid,False));last_scan=time.monotonic();write(out/'status.json',state)
    try:proc.wait(timeout=1)
    except subprocess.TimeoutExpired:pass
   assert proc.returncode==0,str(out/'run.log')
   if preflight:
    paths=sorted({line.split('via ',1)[1].split()[0] for path in out.glob('nccl-*.log') for line in path.read_text().splitlines() if 'via ' in line});assert paths==['SHM/direct/direct'],paths;state['transports']=paths
   else:assert json.loads((out/'phase.json').read_text())['status']=='COMPLETE'
   state['status']='PASS'
  except BaseException as exc:
   state.update(status='FAIL',error=repr(exc))
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  finally:
   state.update(exit_code=proc.returncode,finished_unix=time.time());write(out/'status.json',state);write(PACKET/(label+'_receipt.json'),state);publish('results: OldCA fanout '+label+' '+state['status'])
 return out

def main():
 state=dict(status='RUNNING',stage='PREPARATION',runtime='A3',policies=['BR','CA','OldCA'],horizon=64,tolerance=.001,denominator='BR',started_unix=time.time(),new_trace_capture=False);write(PACKET/'status.json',state)
 try:
  deadline=time.monotonic()+3600
  while not (ROOT/'physical_plans.json').exists():
   if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP')
   meta=json.loads((ROOT/'plan_build_process.json').read_text());proc=Path('/proc')/str(meta['pid'])/'stat'
   if not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[0]=='Z':raise RuntimeError('Offline PLAN builder exited without completed plans; inspect prepare.log')
   if time.monotonic()>deadline:raise TimeoutError('offline preparation deadline')
   time.sleep(5)
  assert json.loads((PACKET/'policy_validation.json').read_text())['status']=='PASS'
  audit_prior_confirmation()
  publish('results: freeze validated common-route decode64 schedules')
  from batch_comm_common import stop_idle_load
  stop_idle_load();deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in h.snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
  for world in [4,8]:
   state.update(stage=f'R{world}_PREFLIGHT');write(PACKET/'status.json',state);run(world,True)
   state.update(stage=f'R{world}_PHYSICAL');write(PACKET/'status.json',state);run(world)
   subprocess.run([PYTHON,str(P/'scripts/summarize_old_ca_fanout.py'),'--world',str(world)],env=env_for(world),check=True);publish('results: summarize R'+str(world)+' OldCA A3 screen')
  state.update(status='COMPLETE',stage='FINISHED',finished_unix=time.time())
 except BaseException as exc:state.update(status='FAILED_OR_STOPPED',error=repr(exc),finished_unix=time.time());raise
 finally:
  state['resident_models']=h.restore();write(PACKET/'status.json',state);publish('results: bounded OldCA fanout checkpoint and GPU handoff')
def audit_prior_confirmation():
 plans=json.loads((ROOT/'physical_plans.json').read_text());audit=[]
 prior=OLD_ROOT/'tolerance_001'
 for plan in plans:
  path=prior/f"R{plan['world']}_physical"/f"{plan['id']}_A3_confirmation.json"
  decision=json.loads(path.read_text()) if path.exists() else None
  reusable=bool(decision and decision['confirm'])
  assert not reusable, 'Reusable confirmation found; implement reuse rather than repeating it'
  plan['prior_confirmation_reusable']=False
  audit.append(dict(workload=plan['id'],path=str(path),prior_decision=decision,reusable=False))
 write(ROOT/'physical_plans.json',plans);write(PACKET/'prior_confirmation_audit.json',audit)

if __name__=='__main__':main()
