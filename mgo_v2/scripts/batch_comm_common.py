"""Bounded batch study paths, exclusive idle-model handoff, and worker guard."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from trace_comm_input import PACKET,sha
from run_rank_oracle_study import memory,process_tree
from run_ipc_baseline_rebase import write
PACKAGE=PACKET.parents[1]
ROOT=Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/batch_comm_20261003')
LOAD=Path('/home/hwlee/mgo-results/model_inference_load_20261003')
WORKER=PACKAGE/'examples/model_inference_load.py'
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'
BATCHES=(4,8,16,32)
ORDER=((0,'T0'),(0,'R3'),(1,'R3'),(1,'T0'))

def owned(pid):
 try:return str(WORKER) in Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
 except FileNotFoundError:return False

def stop_idle_load():
 (LOAD/'STOP').touch()
 ps=json.loads((LOAD/'processes.json').read_text())
 for r in ps:
  if owned(r['pid']):os.kill(r['pid'],signal.SIGTERM)
 deadline=time.monotonic()+20
 while any(owned(r['pid']) for r in ps) and time.monotonic()<deadline:time.sleep(.25)
 for r in ps:
  if owned(r['pid']):os.kill(r['pid'],signal.SIGKILL)
 time.sleep(1)
 assert not any(owned(r['pid']) for r in ps)

def start_idle_load():
 # Never stop or compete with someone else's workload; existing guard also exits on arrival.
 uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 busy={line.split(',')[0].strip() for line in apps.splitlines()}
 (LOAD/'STOP').unlink(missing_ok=True)
 old=json.loads((LOAD/'processes.json').read_text());ps=[r for r in old if owned(r['pid'])]
 restriction=LOAD/'owner_stopped_gpus.json'
 stopped=set(json.loads(restriction.read_text())['gpus']) if restriction.exists() else set()
 for line in uuids.splitlines():
  g,uuid=[x.strip() for x in line.split(',')];g=int(g)
  if g in stopped or uuid in busy or any(r['gpu']==g for r in ps):continue
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  with (LOAD/f'gpu{g}.log').open('a') as f:
   p=subprocess.Popen([PYTHON,'-u',str(WORKER)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  ps.append(dict(gpu=g,pid=p.pid))
 write(LOAD/'processes.json',ps)
 return ps

def run(label,worker,args=(),mode='T0',model=False,smoke=False):
 target=ROOT/label;target.mkdir(exist_ok=False)
 env=dict(os.environ,PYTHONPATH=f'{PACKAGE}:{PACKAGE}/examples',CUDA_VISIBLE_DEVICES='0,1,4,5',OMP_NUM_THREADS='4' if model else '1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MOE_EP_DISABLE_ARCHER_EVICT='1',MOE_EP_NATIVE_NUMERICS='1',MOE_EP_SLOT_VIEWS='1')
 for k in list(env):
  if k.startswith('NCCL_'):del env[k]
 env['NCCL_CUMEM_ENABLE']='0'
 if mode=='R3':env.update(NCCL_P2P_LEVEL='LOC',NCCL_IB_DISABLE='1')
 if smoke:env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(target/'nccl-%h-%p.log'))
 initial=memory();assert initial['host_available_bytes']>=512*2**30 and all(g['used_mib']<1024 for g in initial['gpu'].values()),initial
 cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(PACKAGE/'examples'/worker),'--output',str(target),*map(str,args)]
 limit=600 if model else 180
 state=dict(status='RUNNING',command=cmd,mode=mode,model=model,transport_env={k:v for k,v in env.items() if k.startswith('NCCL_')},started_unix=time.time(),memory=[initial],timeout_seconds=limit)
 write(target/'status.json',state);reason=None
 with (target/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;next_sample=time.monotonic()+5
  try:
   while proc.poll() is None:
    try:proc.wait(timeout=1)
    except subprocess.TimeoutExpired:pass
    if time.time()-state['started_unix']>limit:raise RuntimeError('bounded timeout')
    if proc.poll() is None and time.monotonic()>=next_sample:
     sample=memory();sample['group_rss_bytes']=process_tree(proc.pid)[1];state['memory'].append(sample);write(target/'status.json',state);next_sample=time.monotonic()+5
     if sample['host_available_bytes']<128*2**30 or sample['group_rss_bytes']>(320 if model else 32)*2**30 or any(g['free_mib']<8192 for g in sample['gpu'].values()):raise RuntimeError('memory guard')
  except BaseException as exc:
   reason=repr(exc)
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  finally:
   state.update(status='PASS' if proc.returncode==0 and reason is None else 'FAIL',exit_code=proc.returncode,error=reason,finished_unix=time.time());write(target/'status.json',state)
 assert state['status']=='PASS',str(target/'run.log')
 rs=[json.loads((target/f'rank{r}.json').read_text()) for r in range(4)]
 assert all(r['status']=='PASS' and r['rank']==i for i,r in enumerate(rs))
 if smoke:
  assert all(r['payload_valid'] and r['world_size']==r['local_world_size']==4 for r in rs) and len({r['hostname'] for r in rs})==1
  lines=[l for p in target.glob('nccl-*.log') for l in p.read_text().splitlines() if 'via ' in l]
  paths=sorted({l.split('via ',1)[1].split()[0] for l in lines});assert paths
  assert (paths==['P2P/IPC'] if mode=='T0' else all(t.startswith('SHM/') for t in paths)),paths
  state['transports']=paths;write(target/'status.json',state)
  (PACKET/f'batch_comm_{mode}_transport.log').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(cell=label,status='PASS',seconds=state['finished_unix']-state['started_unix'])),flush=True)
 return rs
