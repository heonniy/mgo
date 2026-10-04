"""Resource-guarded, untimed milestone GPU gate runner."""
import os,json,subprocess,signal,time,argparse
from pathlib import Path
import run_timing_stability as h
from batch_comm_common import stop_idle_load
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004');PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';h.ROOT=ROOT

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',default='M0');args=parser.parse_args();out=ROOT/args.stage;out.mkdir(parents=True,exist_ok=False)
 state=dict(status='RUNNING',stage=args.stage,started_unix=time.time());h.write(PACKET/'status.json',state)
 stop_idle_load();deadline=time.monotonic()+180
 while any(g['temperature_c']>=65 for g in h.snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
 h.safe(h.sample(),True);env=h.env_for('env1');cache=ROOT/'compile_cache';env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
 cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(P/'examples/refactor_baseline_worker.py'),'--inputs',str(ROOT/'baseline_B128'),'--output',str(out)]
 if args.stage in ('M6_NATIVE','M6_PERF'):cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(P/'examples/compact_metadata_check.py'),'--output',str(out)]
 if args.stage.startswith('M8_NCCL'):cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(P/'examples/priority_h2d_collective_check.py'),'--output',str(out)]
 if args.stage=='M6_PERF':cmd+=['--benchmark']
 if args.stage=='M1':cmd+=['--instrument']
 if args.stage=='M5':cmd+=['--arena-budget','2']
 if args.stage=='M6':cmd+=['--arena-budget','2','--compact']
 if args.stage.startswith('M9'):cmd+=['--arena-budget','2','--split-controller','--debug-plan','--physical-prefetch']
 if args.stage.startswith('M10'):cmd+=['--arena-budget','2','--split-controller','--debug-plan','--physical-prefetch','--streaming']
 if args.stage=='M7':cmd+=['--arena-budget','2','--split-controller','--debug-plan']
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;h.write(PACKET/'status.json',state)
  try:
   while proc.poll() is None:
    if time.time()-state['started_unix']>1800:raise TimeoutError('short milestone gate timeout')
    time.sleep(10)
    if proc.poll() is None:h.safe(h.sample(proc.pid))
   assert proc.returncode==0,str(out/'run.log')
   rows=[json.loads((out/f'rank{r}.json').read_text()) for r in range(8)];assert all(r['status']=='PASS' for r in rows)
   state.update(status='PASS',ranks=rows)
  except BaseException as exc:
   state.update(status='FAIL',error=repr(exc))
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  finally:
   state['finished_unix']=time.time();h.write(out/'status.json',state);h.write(PACKET/(args.stage+'_gpu_gate.json'),state);h.write(PACKET/'status.json',{k:v for k,v in state.items() if k!='ranks'})
if __name__=='__main__':main()
