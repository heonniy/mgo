#!/usr/bin/env python3
"""Run replica phase microbench in Env1 and Env2 on four selected GPUs."""
import argparse,json,os,shlex,subprocess,time,signal
from pathlib import Path

P=Path(__file__).resolve().parents[1]

def snapshot(gpus):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True)
 rows={}
 for line in raw.splitlines():
  i,free,temp=map(int,line.split(','));rows[i]=dict(free_mib=free,temp_c=temp)
 return {g:rows[g] for g in gpus}

def run_env(name,gpus,root):
 out=root/name;out.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
          PYTHONPATH=f'{P}:{P}/scripts',NCCL_CUMEM_ENABLE='0')
 for k in list(env):
  if k.startswith('NCCL_') and k!='NCCL_CUMEM_ENABLE':del env[k]
 if name=='env2':env.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
 cmd=['/home/hwlee/sub-moe/phase01/.venv/bin/python','-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
      str(P/'examples/replica_phase_microbench.py'),'--output',str(out),'--environment',name]
 with (out/'run.log').open('w') as f:
  proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (out/'process.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd)))
  try:
   proc.wait(timeout=900)
   if proc.returncode:raise RuntimeError(f'{name} microbench exited {proc.returncode}')
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--root',type=Path,default=Path('/home/hwlee/mgo-results/r8_real_replica_batch_scaling_20261004/microbench'));a=p.parse_args()
 gpus=[int(x) for x in a.gpus.split(',')];assert len(gpus)==4 and len(set(gpus))==4
 initial=snapshot(gpus)
 if any(x['free_mib']<8192 or x['temp_c']>=80 for x in initial.values()):raise RuntimeError(initial)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).strip()
 if apps:print(json.dumps(dict(warning='GPU compute processes exist; ensure selected GPUs are free',apps=apps)),flush=True)
 a.root.mkdir(parents=True,exist_ok=True)
 state=dict(status='RUNNING',gpus=gpus,initial=initial,started_unix=time.time());(a.root/'driver_status.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  for name in ('env1','env2'):run_env(name,gpus,a.root)
  calib=P/'experiments/r8_real_replica_batch_scaling_20261004/microbench_calibration.json'
  subprocess.run(['/home/hwlee/sub-moe/phase01/.venv/bin/python','-u',str(P/'scripts/summarize_replica_phase_microbench.py'),'--root',str(a.root),'--output',str(calib)],check=True)
  state.update(status='PASS',finished_unix=time.time(),calibration=str(calib))
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc),finished_unix=time.time());raise
 finally:(a.root/'driver_status.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':main()
