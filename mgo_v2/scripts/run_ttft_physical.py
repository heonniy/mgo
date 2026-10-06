"""Bounded S1 seed screen then fresh-process S2 best-seed TTFT validation."""
import os,time,subprocess,signal,json,hashlib
from ttft_common import *
from batch_comm_common import stop_idle_load,start_idle_load
import run_timing_stability as h

def publish(message,paths):
 subprocess.run(['git','add',*[str(p) for p in paths]],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  if subprocess.run(['git','push','origin','HEAD:refs/heads/codex/policy-regime-20261005'],cwd=P.parent).returncode:write(ROOT/'publication_pending.json',dict(message=message,unix=time.time()))
def source_identity():
 paths=list((P/'mgo_v2').glob('*.py'))+[P/'examples'/n for n in ['ttft_worker.py','env_offload_worker.py','la_physical_worker.py']]+[P/'scripts'/n for n in ['env_offload_policy.py','env_offload_layout.py','env_offload_tensors.py','br_carep_cpu.py','la_placement.py','old_ca_fanout_policy.py','refactor_thread_affinity.py','ttft_common.py','run_ttft_physical.py']]
 return {str(p.relative_to(P)):sha(p) for p in paths}
def run(label,stage,specs):
 out=ROOT/'physical'/label;out.mkdir(parents=True,exist_ok=True);status=out/'status.json'
 if status.exists():x=json.loads(status.read_text());assert x['status']=='PASS','preserve failed attempt; repair under new label';return out
 write(out/'specs.json',specs);state=dict(status='RUNNING',stage=stage,label=label,started_unix=time.time(),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip(),code=source_identity(),specs=specs,gpus=GPUS,completed=[],boundaries=[])
 def guard(pid=None,initial=False):
  row=h.sample(pid);row['gpus']=[g for g in row['gpus'] if g['gpu'] in GPUS]
  uuids={x.split(',')[1].strip():int(x.split(',')[0]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
  apps=[x.split(',') for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()];target={int(p) for u,p in apps if uuids[u.strip()] in GPUS};row['foreign_pids']=[p for p in row['foreign_pids'] if p in target];h.safe(row,initial)
  if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
  return row
 stop_idle_load();proc=None
 try:
  deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in h.snapshot()['gpus'] if g['gpu'] in GPUS) and time.monotonic()<deadline:time.sleep(5)
  state['initial']=guard(initial=True);env=h.env_for('env1');env.update(CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',TORCHINDUCTOR_CACHE_DIR=str(ROOT/'compile_cache/inductor'),TRITON_CACHE_DIR=str(ROOT/'compile_cache/triton'))
  cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(P/'examples/ttft_worker.py'),'--specs',str(out/'specs.json'),'--output',str(out),'--stage',stage]
  active=None;released=set();last=0
  with (out/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(status,state)
   while proc.poll() is None:
    if time.time()-state['started_unix']>6*3600:raise TimeoutError('bounded6h group')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if active:
     if all((out/f'{active}_measure_rank{r}.json').exists() for r in range(4)):
      state['completed'].append(active);active=None;write(status,state)
    else:
     boundary=json.loads((out/'boundary.json').read_text()) if (out/'boundary.json').exists() else None
     if boundary and boundary['key'] not in released:
      key=boundary['key'];assert all((out/f'{key}_ready_rank{r}.json').exists() for r in range(4));state['boundaries'].append(dict(key=key,resources=guard(proc.pid)));write(status,state);(out/f'{key}_GO').touch();released.add(key);active=key
     elif time.monotonic()-last>30:guard(proc.pid);last=time.monotonic()
    time.sleep(1)
   assert proc.returncode==0,str(out/'run.log')
  assert state['code']==source_identity(),'source changed during physical run';result=json.loads((out/'result.json').read_text());assert result['status']=='PASS';state.update(status='PASS',results=result,completed=sorted(released))
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  state['finished_unix']=time.time();write(status,state);receipt=PACKET/(label+'.json');write(receipt,state);write(ROOT/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()));publish(f'TTFT: {label} {state["status"]}',[receipt])
 return out

def main():
 assert (ROOT/'cpu_ready').exists()
 rows=json.loads((PACKET/'TTFT_S0_SHORTLIST.json').read_text())['rows'];state=dict(status='RUNNING',stage='S1',completed=[],selected=[],started_unix=time.time());write(ROOT/'physical_status.json',state)
 try:
  for cache,batch in [(30,16),(60,16),(30,128),(60,128)]:
   candidates={p:[r for r in rows if (r['cache'],r['batch'],r['policy'])==(cache,batch,p)] for p in ('CA','OLD_CA','LA')};specs=[]
   for i in range(8):
    for p in candidates:
     r=candidates[p][i];w=r['workload_seed'];s=r['placement_seed'];specs.append(dict(key=f'C{cache}_B{batch}_{p}_w{w}_p{s}',cache=cache,batch=batch,policy=p,workload_seed=w,placement_seed=s,order=len(specs)%2,inputs=str(ROOT/'inputs'/f'C{cache}_B{batch}_w{w}_p{s}')))
   out=run(f'TTFT_S1_C{cache}_B{batch}','S1',specs);state['completed'].append(str(out));write(ROOT/'physical_status.json',state)
   for policy in candidates:
    valid=[json.loads(f.read_text()) for f in out.glob('*_result.json')];valid=[r for r in valid if r['status']=='PASS' and r['spec']['policy']==policy]
    if not valid:state.setdefault('invalid_regimes',[]).append(dict(cache=cache,batch=batch,policy=policy));continue
    winner=max(valid,key=lambda r:r['gain']);state['selected'].append(winner['spec'])
  state['stage']='S2';write(ROOT/'physical_status.json',state);write(PACKET/'TTFT_SELECTED_SEEDS.json',dict(status='PASS',selected=state['selected'],invalid_regimes=state.get('invalid_regimes',[])));publish('TTFT: freeze physically selected best seeds',[PACKET/'TTFT_SELECTED_SEEDS.json'])
  for spec in state['selected']:
   out=run('TTFT_S2_'+spec['key'],'S2',[spec]);state['completed'].append(str(out));write(ROOT/'physical_status.json',state)
  state['status']='PASS'
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:state['finished_unix']=time.time();write(ROOT/'physical_status.json',state);write(PACKET/'TTFT_PHYSICAL_EXECUTION.json',state)
 if state['status']=='PASS':
  import report_ttft
  report_ttft.main();publish('results: fixed512 ShareGPT best-seed TTFT headroom',[PACKET/'TTFT_PHYSICAL_EXECUTION.json',PACKET/'TTFT_RESULTS.json',PACKET/'TTFT_RESULTS.md'])
if __name__=='__main__':main()
