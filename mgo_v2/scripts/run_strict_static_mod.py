"""Append eight STATIC_MOD baseline cells after active final validation."""
import os,time,subprocess,signal,json
from pathlib import Path
from strict_headroom_common import *
from strict_headroom_guard import guard,cooldown
from run_strict_headroom_pipeline import publish
from batch_comm_common import stop_idle_load,start_idle_load
import run_timing_stability as h

def run_cell(row):
 world=row['world'];gpus=R4_GPUS if world==4 else list(range(8));label=f'STATIC32_R{world}_B{row["batch"]}_L{row["context"]}'
 out=ROOT/'physical'/label;out.mkdir(exist_ok=False)
 path=ROOT/'static_inputs'/f'R{world}_B{row["batch"]}_L{row["context"]}'
 reference=ROOT/'physical'/f'S2D32_R{world}_B{row["batch"]}_L{row["context"]}'/'result.json'
 spec=dict(row,inputs=str(path),reference_result=str(reference),order=0);write(out/'specs.json',[spec])
 state=dict(status='RUNNING',spec=spec,world=world,gpus=gpus,started_unix=time.time(),initial=cooldown(gpus),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip())
 paths=list((P/'mgo_v2').glob('*.py'))+[P/'examples/strict_static_mod_worker.py',P/'examples/la_physical_worker.py',P/'scripts/env_offload_policy.py',P/'scripts/la_placement.py',P/'scripts/strict_static_mod_policy.py',P/'examples/strict_decode32_worker.py'];state['code']={str(p):sha(p) for p in paths}
 env=h.env_for('env1');env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),MGO_V2_PHYSICAL_GPUS=','.join(map(str,gpus)))
 cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'examples/strict_static_mod_worker.py'),'--specs',str(out/'specs.json'),'--output',str(out)]
 proc=None;active=None;released=set();last=0
 try:
  with (out/'run.log').open('w') as log:
   proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(out/'status.json',state)
   while proc.poll() is None:
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if time.time()-state['started_unix']>4*3600:raise TimeoutError('bounded4h cell')
    if active and all((out/f'{active}_measure_rank{r}.json').exists() for r in range(world)):active=None
    if active is None:
     boundary=json.loads((out/'boundary.json').read_text()) if (out/'boundary.json').exists() else None
     if boundary and boundary['key'] not in released:
      key=boundary['key'];assert all((out/f'{key}_ready_rank{r}.json').exists() for r in range(world));state.setdefault('boundaries',[]).append(dict(key=key,resources=guard(gpus,proc.pid)));write(out/'status.json',state);(out/f'{key}_GO').touch();released.add(key);active=key
     elif time.monotonic()-last>30:guard(gpus,proc.pid);last=time.monotonic()
    time.sleep(1)
   assert proc.returncode==0,str(out/'run.log')
  assert all(sha(p)==v for p,v in state['code'].items())
  result=json.loads((out/'result.json').read_text());assert result['status']=='PASS';state.update(status='PASS',result=result)
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  state['finished_unix']=time.time();write(out/'status.json',state);receipt=PACKET/(label+'.json');write(receipt,state);publish('static baseline: '+label+' '+state['status'],[receipt])
 return state

def main():
 state=dict(status='RUNNING',stage='WAIT_DECODE32_COMPLETE',completed=[],started_unix=time.time())
 def save():write(ROOT/'static_mod_status.json',state)
 save()
 try:
  deadline=time.monotonic()+12*3600
  while True:
   old=json.loads((ROOT/'decode32_status.json').read_text())
   if old['status']=='PASS':break
   assert old['status']=='RUNNING',old.get('error')
   if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
   if time.monotonic()>deadline:raise TimeoutError('bounded previous queue wait')
   time.sleep(5)
  # Wait for the prior queue's restore/publication to finish.
  end=time.monotonic()+180
  while Path('/proc/2483080/cmdline').exists() and Path('/proc/2483080/cmdline').read_bytes():
   if time.monotonic()>end:raise TimeoutError('previous queue finalization')
   time.sleep(1)
  state['stage']='CPU_STATIC_PROOFS';save()
  with (ROOT/'static_cpu.log').open('w') as f:subprocess.run([h.PYTHON,'-u',str(P/'scripts/prepare_strict_static_mod.py')],env=h.env_for('env1'),stdout=f,stderr=subprocess.STDOUT,check=True)
  publish('static baseline: fixed-owner CPU proofs',[PACKET/'STATIC_MOD_INPUTS.json'])
  selected=[x['row'] for x in json.loads((PACKET/'DECODE32_SELECTED.json').read_text())['selected']]
  stop_idle_load();state['stage']='STATIC_PRIMARY';save()
  for row in selected:
   result=run_cell(row);state['completed'].append(result['spec']);save()
  import report_strict_static_mod
  report_strict_static_mod.main();publish('results: static owner TTFT TPOT E2E baseline',[PACKET/'STATIC_MOD_RESULTS.json',PACKET/'STATIC_MOD_RESULTS.md']);state['status']='PASS'
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();save();write(PACKET/'STATIC_MOD_EXECUTION.json',state);write(ROOT/'resident_models.json',dict(processes=start_idle_load()));publish('static baseline pipeline '+state['status'],[PACKET/'STATIC_MOD_EXECUTION.json'])
if __name__=='__main__':main()
