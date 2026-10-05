"""Run only checkpoint A, hand GPUs back on success/failure, never queue B."""
import json,os,signal,subprocess,time
from pathlib import Path
import run_timing_stability as h
from batch_comm_common import stop_idle_load,start_idle_load
from prepare_critical_microbench import ROOT,PACKET,P,write

def main():
 assert not (ROOT/'attempt1').exists(),'preserve attempts; explicit recovery required'
 out=ROOT/'attempt1';out.mkdir();state=dict(status='PREPARING',started_unix=time.time(),plan_commit='894ed896292d3ff628f3db2dbc34a04d45461393')
 proc=None
 try:
  stop_idle_load()
  # Bounded cool-down; no timing until the same initial shared-resource gates pass.
  for i in range(13):
   sample=h.sample()
   if all(g['temperature_c']<65 for g in sample['gpus']):break
   time.sleep(5)
  h.safe(sample,True);state['initial_resources']=sample
  env=h.env_for('env1');env.update(CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',MGO_V2_NUMA_STRICT='0',OMP_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(out/'nccl-%h-%p.log'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'compile_cache/inductor'),TRITON_CACHE_DIR=str(ROOT/'compile_cache/triton'))
  cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(P/'examples/critical_microbench_worker.py'),'--inputs',str(ROOT/'inputs.json'),'--output',str(out)]
  state.update(status='RUNNING',command=cmd,transport_env={k:v for k,v in env.items() if k.startswith('NCCL_')},source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip());write(ROOT/'status.json',state)
  with (out/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(ROOT/'status.json',state)
   while proc.poll() is None:
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:pass
    if (ROOT/'STOP').exists():raise RuntimeError('owner stop')
    if time.time()-state['started_unix']>3600:raise TimeoutError('one-hour bounded checkpoint A')
    # Small model-free arrays, capped CUDA allocator; boundary-independent safety
    # scan at 15-second cadence is recorded and identical for all conditions.
    if proc.poll() is None:h.safe(h.sample(proc.pid))
   assert proc.returncode==0,str(out/'run.log')
  ranks=[json.loads((out/f'rank{r}.json').read_text()) for r in range(4)];assert all(x['status']=='PASS' for x in ranks)
  lines=[l for p in out.glob('nccl-*.log') for l in p.read_text().splitlines() if 'via ' in l]
  paths=sorted({l.split('via ',1)[1].split()[0] for l in lines});assert paths==['P2P/IPC'],paths
  state.update(status='MEASUREMENTS_COMPLETE',transport_paths=paths)
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc))
  if proc and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise
 finally:
  state['finished_unix']=time.time();write(ROOT/'status.json',state);write(PACKET/'EXECUTION_STATUS.json',state)
  write(ROOT/'restored_models.json',dict(unix=time.time(),processes=start_idle_load()))
if __name__=='__main__':main()
