"""Guarded four-GPU capture; never touch other owners' GPUs or processes."""
import os,time,subprocess,signal,json
from ttft_common import *
from batch_comm_common import stop_idle_load,start_idle_load
import run_timing_stability as h

def main():
 assert json.loads((ROOT/'requests.json').read_text())['status']=='PASS'
 assert not (ROOT/'capture_status.json').exists(),'preserve prior attempts'
 state=dict(status='RUNNING',started_unix=time.time(),gpus=GPUS,workers=[]);workers=[];logs=[]
 stop_idle_load()
 def guard():
  snap=h.sample();snap['gpus']=[g for g in snap['gpus'] if g['gpu'] in GPUS]
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines();mapping={r.split(',')[1].strip():int(r.split(',')[0]) for r in rows}
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
  for line in apps:
   uuid,pid=line.split(',');assert mapping[uuid.strip()] not in GPUS or int(pid) in {p.pid for p in workers},'foreign process on assigned GPU'
  snap['foreign_pids']=[];h.safe(snap);return snap
 try:
  deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in h.snapshot()['gpus'] if g['gpu'] in GPUS) and time.monotonic()<deadline:time.sleep(5)
  state['initial']=guard()
  for gpu in GPUS:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');log=(ROOT/f'capture_gpu{gpu}.log').open('w');logs.append(log)
   proc=subprocess.Popen([h.PYTHON,'-u',str(P/'examples/ttft_capture.py'),'--gpu',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);workers.append(proc);state['workers'].append(dict(gpu=gpu,pid=proc.pid))
  while any(p.poll() is None for p in workers):
   assert all(p.poll() in (None,0) for p in workers),'capture worker failed'
   if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
   if time.time()-state['started_unix']>7200:raise TimeoutError('bounded2h capture')
   state['resources']=guard();write(ROOT/'capture_status.json',state);time.sleep(5)
  assert all(p.returncode==0 for p in workers)
  state['receipts']=[json.loads((ROOT/'capture'/f'gpu{g}'/'receipt.json').read_text()) for g in GPUS];assert all(r['status']=='PASS' for r in state['receipts']);state['status']='PASS'
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  for p in workers:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  for p in workers:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  for log in logs:log.close()
  state['finished_unix']=time.time();write(ROOT/'capture_status.json',state);write(PACKET/'TTFT_CAPTURE.json',state);write(ROOT/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':main()
