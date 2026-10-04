"""Bounded R8 LOAD BR/LA physical queue; boundary-only monitoring in measurements."""
import os,json,subprocess,signal,time,statistics
from pathlib import Path
import run_timing_stability as h
from batch_comm_common import stop_idle_load
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/la_physical_validation_20261004');PACKET=P/'experiments/la_physical_validation_20261004'
h.ROOT=ROOT
BRANCH='codex/r8-b128-b256-real-replica-20261004'
write=h.write

def publish(message):
 subprocess.run(['git','add',str(PACKET)],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  if subprocess.run(['git','push','origin','HEAD:'+BRANCH],cwd=P.parent).returncode:
   write(ROOT/'publication_pending.json',dict(message=message,unix=time.time()))

def run(batch,policy):
 out=ROOT/f'B{batch}_{policy}_env1'
 if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['status']=='PASS':return
 out.mkdir(exist_ok=True)
 s=dict(status='RUNNING',batch=batch,policy=policy,started_unix=time.time(),monitor='resource scans only outside timed region')
 before=h.sample();h.safe(before,True);s['initial']=before
 env=h.env_for('env1');cache=ROOT/'compile_cache';env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
 cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(P/'examples/la_physical_worker.py'),'--inputs',str(ROOT/f'B{batch}'),'--policy',policy,'--output',str(out)]
 active=None;released=set();last_scan=0
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);s['pid']=proc.pid;write(out/'status.json',s)
  try:
   while proc.poll() is None:
    if time.time()-s['started_unix']>4*3600:raise TimeoutError('four-hour bounded workload/policy phase')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if active is not None:
     if len(list(out.glob(f'measure_{active}_rank*.json')))==8:
      s.setdefault('completed_repeats',[]).append(active);write(out/'status.json',s)
      write(PACKET/f'B{batch}_{policy}_repeat{active}.json',dict(status='PASS',ranks=[json.loads((out/f'measure_{active}_rank{r}.json').read_text()) for r in range(8)]))
      publish(f'results: R8 B{batch} {policy} physical repeat {active}')
      active=None
    else:
     ready=next((r for r in (1,2,3) if r not in released and len(list(out.glob(f'ready_{r}_rank*.json')))==8),None)
     if ready is not None:
      boundary=h.sample(proc.pid,True);h.safe(boundary)
      s.setdefault('boundaries',[]).append(dict(repeat=ready,sample=boundary));write(out/'status.json',s)
      (out/f'GO_{ready}').touch();released.add(ready);active=ready
     elif time.monotonic()-last_scan>30:
      sample=h.sample(proc.pid);h.safe(sample);last_scan=time.monotonic()
    time.sleep(2)
   assert proc.returncode==0,str(out/'run.log')
   result=json.loads((out/'result.json').read_text());assert result['status']=='PASS'
   s.update(status='PASS',result=result)
  except BaseException as exc:
   s.update(status='FAIL',error=repr(exc))
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  finally:
   # Reap torchrun children before handing GPUs back.
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   s.update(exit_code=proc.returncode,finished_unix=time.time());write(out/'status.json',s);write(PACKET/f'B{batch}_{policy}_status.json',s);publish(f'results: R8 B{batch} {policy} physical {s["status"]}')

def summarize():
 result={};lines=['# R8 LOAD BR vs LA physical validation','','Frozen exact routes and teacher tokens; live placement; pinned H2D; fetch barrier; Env1.','E2E includes prefill + decode256. TPOT covers decode. Values are seconds.','']
 for batch in (128,256):
  pair={}
  for policy in ('BR','LA'):
   r=json.loads((ROOT/f'B{batch}_{policy}_env1/result.json').read_text());rows=r['samples']
   pair[policy]=dict(unstable=r['unstable'],gate=r['gate'],samples=rows,summary={k:dict(mean=statistics.mean(x[k] for x in rows),median=statistics.median(x[k] for x in rows),min=min(x[k] for x in rows),max=max(x[k] for x in rows)) for k in ('E2E_wall','TPOT')})
  pair['gain']={k:1-pair['LA']['summary'][k]['median']/pair['BR']['summary'][k]['median'] for k in ('E2E_wall','TPOT')}
  pair['primary_comparison_valid']=not any(pair[p]['unstable'] for p in ('BR','LA'))
  result[str(batch)]=pair
  lines.append(f'- B{batch}: E2E gain {pair["gain"]["E2E_wall"]:.2%}; TPOT gain {pair["gain"]["TPOT"]:.2%}; stable comparison={pair["primary_comparison_valid"]}.')
 write(PACKET/'RESULT.json',dict(status='PASS',batches=result));(PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
 state=dict(status='RUNNING',stage='GPU_PREFLIGHT',gpus=list(range(8)),started_unix=time.time());write(PACKET/'status.json',state)
 try:
  for batch in (128,256):assert json.loads((ROOT/f'B{batch}/receipt.json').read_text())['status']=='PASS'
  stop_idle_load();deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in h.snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
  h.safe(h.sample(),True)
  for batch in (128,256):
   for policy in ('BR','LA'):
    state.update(stage=f'B{batch}_{policy}_env1');write(PACKET/'status.json',state);run(batch,policy)
  summarize();state.update(status='COMPLETE',stage='FINISHED')
 except BaseException as exc:
  state.update(status='FAILED',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();write(PACKET/'status.json',state);publish('results: R8 LA physical checkpoint')
  state['resident_models']=h.restore();write(PACKET/'status.json',state);publish('ops: restore model load after LA physical validation')
if __name__=='__main__':main()
