"""Guarded fresh-process phase launcher. Does not expand the frozen matrix."""
import argparse,json,os,signal,subprocess,time,hashlib
from pathlib import Path
from run_rank_oracle_study import process_tree
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003');PACKET=P/'experiments/env_e2e_tpot_offload_20261003'
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def snapshot():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True)
 return dict(gpus=[dict(zip(['gpu','used_mib','free_mib','temperature_c'],map(int,r.split(',')))) for r in rows.splitlines()],host_available_bytes=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
def run(cell,policy,phase,environment,repeat=0,comm_mode='current'):
 if comm_mode=='coslot' and phase=='PLAN':raise ValueError('CoSLoT remeasure must reuse the validated frozen PLAN')
 spec=json.loads((ROOT/'frozen_matrix.json').read_text())['cells'][cell];world=spec['ranks'];gpus=list(range(8)) if world==8 else [0,1,4,5]
 suffix='' if comm_mode=='current' else f'_{comm_mode}';label=f'{cell}_{policy}_{environment}_{phase}{suffix}_{repeat}';out=ROOT/label;out.mkdir(exist_ok=False);state=dict(status='RUNNING',cell=cell,policy=policy,phase=phase,environment=environment,repeat=repeat,started_unix=time.time(),samples=[])
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
 cache=ROOT/'compile_cache'/f'{cell}_{policy}_{environment}_{comm_mode}';cache.mkdir(parents=True,exist_ok=True)
 env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
 cmd=['/home/hwlee/sub-moe/phase01/.venv/bin/python','-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'examples/env_offload_worker.py'),'--cell',cell,'--policy',policy,'--phase',phase,'--environment',environment,'--comm-mode',comm_mode,'--output',str(out)]
 if phase!='PLAN':
  plans=[p for p in ROOT.glob(f'{cell}_{policy}_env1_PLAN_*') if (p/'status.json').exists() and json.loads((p/'status.json').read_text())['status']=='PASS']
  assert len(plans)==1,plans
  cmd+=['--plan',str(plans[0])]
 state.update(comm_mode=comm_mode,command=cmd,source_sha256=hashlib.sha256((P/'examples/env_offload_worker.py').read_bytes()).hexdigest(),policy_sha256=hashlib.sha256((P/'scripts/env_offload_policy.py').read_bytes()).hexdigest(),layout_sha256=hashlib.sha256((P/'scripts/env_offload_layout.py').read_bytes()).hexdigest(),transport_env={k:v for k,v in env.items() if k.startswith('NCCL_')},compile_cache=str(cache),initial=initial)
 (out/'worker_source.py').write_bytes((P/'examples/env_offload_worker.py').read_bytes());(out/'policy_source.py').write_bytes((P/'scripts/env_offload_policy.py').read_bytes());(out/'layout_source.py').write_bytes((P/'scripts/env_offload_layout.py').read_bytes())
 reason=None
 with (out/'run.log').open('w') as f:
  proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(out/'status.json',state)
  while proc.poll() is None:
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
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--policy',required=True);p.add_argument('--phase',required=True);p.add_argument('--environment',default='env1');p.add_argument('--repeat',type=int,default=0);p.add_argument('--comm-mode',choices=['current','coslot'],default='current');a=p.parse_args();run(a.cell,a.policy,a.phase,a.environment,a.repeat,a.comm_mode)
