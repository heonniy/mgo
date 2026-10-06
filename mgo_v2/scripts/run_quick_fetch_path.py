"""Guarded quick runner for R4 fetch-path microbenchmark.

Uses physical GPUs 0,1,4,5 only. Does not scan/kill processes on 2,3,6,7.
"""
import os,time,subprocess,signal,json
from pathlib import Path

P=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MGO_QUICK_FETCH_ROOT','/home/hwlee/mgo-results/quick_fetch_path_20261006'))
PACKET=P/'experiments/quick_fetch_path_20261006'
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'
GPUS=[0,1,4,5]

def write(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)

def target_gpu_state():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=[]
 for line in out.splitlines():
  g,mem,temp,util=[int(float(x.strip())) for x in line.split(',')]
  if g in GPUS:rows.append(dict(gpu=g,memory_used_mib=mem,temperature_c=temp,utilization_gpu=util))
 assert sorted(x['gpu'] for x in rows)==GPUS
 return rows

def foreign_target_processes():
 uuid_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
 mapping={u.strip():int(g.strip()) for g,u in (x.split(',') for x in uuid_rows)}
 app_rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
 return [dict(gpu=mapping[u.strip()],pid=int(pid.strip())) for u,pid in (x.split(',') for x in app_rows) if mapping[u.strip()] in GPUS]

def main():
 assert not ROOT.exists(),'preserve prior attempt; use a new label if rerunning'
 ROOT.mkdir(parents=True);PACKET.mkdir(parents=True,exist_ok=True)
 state=dict(status='RUNNING',physical_gpus=GPUS,started_unix=time.time(),initial=target_gpu_state())
 foreign=foreign_target_processes()
 if foreign:raise RuntimeError(f'foreign process on owned target GPU(s): {foreign}')
 # Keep this microbench standalone: no model process manipulation and never touch 2,3,6,7.
 env=dict(os.environ,PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P/"scripts"}:{P/"examples"}',
  CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',
  OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 # Use the previously established IPC baseline; NCCL is only for barriers.
 for key in list(env):
  if key.startswith('NCCL_'):del env[key]
 env.update(NCCL_CUMEM_ENABLE='0',NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH',NCCL_DEBUG_FILE=str(ROOT/'nccl-%h-%p.log'))
 state['transport_env']={k:v for k,v in env.items() if k.startswith('NCCL_')}
 cmd=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
  str(P/'examples/quick_fetch_path_worker.py'),'--output',str(ROOT)]
 state['command']=cmd;write(ROOT/'status.json',state)
 proc=None
 with (ROOT/'run.log').open('w') as log:
  try:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state['pid']=proc.pid;write(ROOT/'status.json',state)
   while proc.poll() is None:
    if time.time()-state['started_unix']>600:raise TimeoutError('quick fetch benchmark exceeded 10 minutes')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    time.sleep(2)
   assert proc.returncode==0,str(ROOT/'run.log')
   result=json.loads((ROOT/'summary.json').read_text());assert result['status']=='PASS'
   state.update(status='PASS',result=result)
  except BaseException as exc:
   state.update(status='FAIL',error=repr(exc))
   if proc is not None and proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  finally:
   state['finished_unix']=time.time();state['final']=target_gpu_state()
   write(ROOT/'status.json',state);write(PACKET/'EXECUTION.json',state)
 print(json.dumps(state.get('result',{}),indent=2))

if __name__=='__main__':main()
