"""Guarded fresh-process phase launcher. Does not expand the frozen matrix."""
import argparse,json,os,signal,subprocess,time,hashlib
from pathlib import Path
from run_rank_oracle_study import process_tree
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003');PACKET=P/'experiments/env_e2e_tpot_offload_20261003'
def discover_validated_plan(cell,policy):
 candidates=[]
 for path in ROOT.glob(f'{cell}_{policy}_*_PLAN_*'):
  try:status=json.loads((path/'status.json').read_text());validation=json.loads((path/'schedule_validation.json').read_text())
  except FileNotFoundError:continue
  if status.get('status')=='PASS' and validation.get('status')=='PASS':candidates.append(path)
 if not candidates:return None
 candidates.sort(key=lambda p:p.stat().st_mtime,reverse=True);return candidates[0]
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def snapshot():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True)
 return dict(gpus=[dict(zip(['gpu','used_mib','free_mib','temperature_c'],map(int,r.split(',')))) for r in rows.splitlines()],host_available_bytes=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
def run(cell,policy,phase,environment,repeat=0,comm_mode='current',h2d_mode='pageable',plan_path=None,h2d_stages=2):
 if phase=='PLAN' and comm_mode!='current':raise ValueError('reference PLAN only uses current transport')
 spec=json.loads((ROOT/'frozen_matrix.json').read_text())['cells'][cell];world=spec['ranks'];gpus=list(range(8)) if world==8 else [0,1,4,5]
 tags=[]
 if comm_mode!='current':tags.append(comm_mode)
 if h2d_mode!='pageable':tags.append(h2d_mode)
 suffix='' if not tags else '_'+('_'.join(tags));label=f'{cell}_{policy}_{environment}_{phase}{suffix}_{repeat}';out=ROOT/label;out.mkdir(exist_ok=False);state=dict(status='RUNNING',cell=cell,policy=policy,phase=phase,environment=environment,repeat=repeat,started_unix=time.time(),samples=[])
 initial=snapshot();cooldown=time.monotonic()
 while any(g['temperature_c']>=65 for g in initial['gpus']) and time.monotonic()-cooldown<180:
  time.sleep(5);initial=snapshot()
 assert initial['host_available_bytes']>=768*2**30 and all(g['free_mib']>76000 and g['temperature_c']<65 for g in initial['gpus'])
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 env=dict(os.environ,PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P}/scripts',CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2')
 for k in list(env):
  if k.startswith('NCCL_'):del env[k]
 env['NCCL_CUMEM_ENABLE']='0'
 if environment=='env2':env.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
 cache=ROOT/'compile_cache'/f'{cell}_{policy}_{environment}_{comm_mode}_{h2d_mode}';cache.mkdir(parents=True,exist_ok=True)
 env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
 cmd=['/home/hwlee/sub-moe/phase01/.venv/bin/python','-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'examples/env_offload_worker.py'),'--cell',cell,'--policy',policy,'--phase',phase,'--environment',environment,'--comm-mode',comm_mode,'--h2d-mode',h2d_mode,'--h2d-stages',str(h2d_stages),'--output',str(out)]
 if phase!='PLAN':
  plan_path=Path(plan_path) if plan_path is not None else discover_validated_plan(cell,policy)
  if plan_path is None:raise FileNotFoundError(f'no validated PLAN for {cell}/{policy}; use ensure_plan()')
  assert json.loads((plan_path/'schedule_validation.json').read_text())['status']=='PASS'
  cmd+=['--plan',str(plan_path)]
 state.update(comm_mode=comm_mode,h2d_mode=h2d_mode,h2d_stages=h2d_stages,plan_path=(str(plan_path) if phase!='PLAN' else None),command=cmd,source_sha256=hashlib.sha256((P/'examples/env_offload_worker.py').read_bytes()).hexdigest(),policy_sha256=hashlib.sha256((P/'scripts/env_offload_policy.py').read_bytes()).hexdigest(),layout_sha256=hashlib.sha256((P/'scripts/env_offload_layout.py').read_bytes()).hexdigest(),transport_env={k:v for k,v in env.items() if k.startswith('NCCL_')},compile_cache=str(cache),initial=initial)
 (out/'worker_source.py').write_bytes((P/'examples/env_offload_worker.py').read_bytes());(out/'policy_source.py').write_bytes((P/'scripts/env_offload_policy.py').read_bytes());(out/'layout_source.py').write_bytes((P/'scripts/env_offload_layout.py').read_bytes())
 reason=None
 with (out/'run.log').open('w') as f:
  proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(out/'status.json',state)
  timed=False
  while proc.poll() is None:
   if phase=='MEASURE' and (comm_mode!='current' or h2d_mode=='pinned') and len(list(out.glob('ready_rank*.json')))==world:
    boundary=snapshot();assert boundary['host_available_bytes']>=256*2**30 and all(g['free_mib']>8192 and g['temperature_c']<85 for g in boundary['gpus'])
    state['before_measure']=boundary;state['GO_unix']=time.time();write(out/'status.json',state);(out/'GO').touch();timed=True
    while proc.poll() is None:
     if (ROOT/'STOP').exists() or time.time()-state['GO_unix']>3600:
      reason='owner_stop_or_timing_timeout';os.killpg(proc.pid,signal.SIGTERM)
      try:proc.wait(timeout=15)
      except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
      break
     try:proc.wait(timeout=1)
     except subprocess.TimeoutExpired:pass
    state['after_measure']=snapshot()
    if any(g['temperature_c']>=85 or g['free_mib']<8192 for g in state['after_measure']['gpus']):reason='post_measure_guard'
    break
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:pass
   if proc.poll() is not None:break
   sample_started=time.monotonic();sample=snapshot();pids,rss=process_tree(proc.pid);pss=0
   for pid in pids:
    try:pss+=next(int(l.split()[1])*1024 for l in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines() if l.startswith('Pss:'))
    except (FileNotFoundError,ProcessLookupError):pass
   sample.update(unix=time.time(),rss_bytes=rss,pss_bytes=pss,collection_seconds=time.monotonic()-sample_started);state['samples'].append(sample);write(out/'status.json',state)
   if sample['host_available_bytes']<256*2**30 or pss>768*2**30 or any(g['free_mib']<8192 or g['temperature_c']>=85 for g in sample['gpus']):reason='memory_or_temperature_guard'
   if (ROOT/'STOP').exists():reason='owner_stop'
   if time.time()-state['started_unix']>7200:reason='bounded_2h_phase_timeout'
   pids,_=process_tree(proc.pid)
   current=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
   if any(int(pid) not in pids and int(pid)!=proc.pid for pid in current.splitlines()):reason='foreign_gpu_process'
   if reason:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  state.update(status='PASS' if proc.returncode==0 and reason is None else 'FAIL',exit_code=proc.returncode,stop_reason=reason,finished_unix=time.time());write(out/'status.json',state)
 receipt={k:v for k,v in state.items() if k!='samples'};receipt['peak_rss_bytes']=max((s['rss_bytes'] for s in state['samples']),default=0);receipt['peak_pss_bytes']=max((s.get('pss_bytes',0) for s in state['samples']),default=0);receipt['raw_root']=str(out)
 (PACKET/'phase_receipts').mkdir(exist_ok=True);write(PACKET/'phase_receipts'/f'{label}.json',receipt);print(json.dumps(receipt),flush=True)
 if state['status']!='PASS':raise RuntimeError(str(out/'run.log'))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--policy',required=True);p.add_argument('--phase',required=True);p.add_argument('--environment',default='env1');p.add_argument('--repeat',type=int,default=0);p.add_argument('--comm-mode',choices=['current','coslot','coslot-active'],default='current');p.add_argument('--h2d-mode',choices=['pageable','pinned'],default='pageable');p.add_argument('--h2d-stages',type=int,default=2);p.add_argument('--plan',type=Path);a=p.parse_args();run(a.cell,a.policy,a.phase,a.environment,a.repeat,a.comm_mode,a.h2d_mode,a.plan,a.h2d_stages)
