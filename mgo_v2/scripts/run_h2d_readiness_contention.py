"""Run the owner-authorized R4 H2D readiness/contention characterization.

Targets physical GPUs 0,1,4,5 only. Other GPUs are ignored and never modified.
"""
import os,time,subprocess,signal,json
from pathlib import Path
from batch_comm_common import stop_idle_load,start_idle_load
from ttft_common import P,PACKET,ROOT,GPUS,write,sha
import run_timing_stability as h

LABEL='H2D_READINESS_CONTENTION_R4'
OUT=ROOT/'h2d_readiness_contention'

def publish(message,paths):
 subprocess.run(['git','add',*[str(p) for p in paths]],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  if subprocess.run(['git','push','origin','HEAD:refs/heads/codex/policy-regime-20261005'],cwd=P.parent).returncode:
   write(ROOT/'publication_pending.json',dict(message=message,unix=time.time()))

def target_sample(pid=None):
 row=h.sample(pid);row['gpus']=[g for g in row['gpus'] if g['gpu'] in GPUS]
 uuids={x.split(',')[1].strip():int(x.split(',')[0]) for x in subprocess.check_output(
  ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
 apps=[x.split(',') for x in subprocess.check_output(
  ['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()]
 target={int(p) for u,p in apps if uuids[u.strip()] in GPUS}
 row['foreign_pids']=[p for p in row['foreign_pids'] if p in target]
 return row

def main():
 assert GPUS==[0,1,4,5]
 if (OUT/'status.json').exists():
  prior=json.loads((OUT/'status.json').read_text())
  assert prior['status']!='PASS','completed result already exists'
 OUT.mkdir(parents=True,exist_ok=True)
 worker=P/'examples/h2d_readiness_contention_worker.py'
 state=dict(status='RUNNING',label=LABEL,physical_gpus=GPUS,started_unix=time.time(),
            source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip(),
            code={str(worker.relative_to(P)):sha(worker),str((P/'mgo_v2/pinned_h2d.py').relative_to(P)):sha(P/'mgo_v2/pinned_h2d.py')})
 stop_idle_load();proc=None
 try:
  deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in h.snapshot()['gpus'] if g['gpu'] in GPUS) and time.monotonic()<deadline:time.sleep(5)
  initial=target_sample();h.safe(initial,True);state['initial']=initial
  env=h.env_for('env1');env.update(CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',MGO_V2_NUMA_STRICT='1')
  cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(worker),'--output',str(OUT)]
  state['command']=cmd;write(OUT/'status.json',state)
  with (OUT/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(OUT/'status.json',state)
   last=time.monotonic()
   while proc.poll() is None:
    if time.time()-state['started_unix']>1800:raise TimeoutError('bounded 30min H2D characterization')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if time.monotonic()-last>10:
     row=target_sample(proc.pid);h.safe(row);state.setdefault('resources',[]).append(row);write(OUT/'status.json',state);last=time.monotonic()
    time.sleep(1)
   assert proc.returncode==0,str(OUT/'run.log')
  rows=[json.loads((OUT/f'rank{r}.json').read_text()) for r in range(4)]
  assert all(x['status']=='PASS' and x['physical_gpu']==GPUS[i] for i,x in enumerate(rows))
  summary=json.loads((OUT/'summary.json').read_text());assert summary['status']=='PASS'
  state.update(status='PASS',summary=summary)
 except BaseException as exc:
  state.update(status='FAIL',error=repr(exc));raise
 finally:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  state['finished_unix']=time.time();write(OUT/'status.json',state)
  receipt=PACKET/'H2D_READINESS_EXECUTION.json';write(receipt,state)
  write(ROOT/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
  publish(f'H2D readiness: {state["status"]}',[receipt])
 if state['status']=='PASS':
  print(json.dumps(state['summary'],indent=2))

if __name__=='__main__':main()
