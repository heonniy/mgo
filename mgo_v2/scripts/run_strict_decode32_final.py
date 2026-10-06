"""Resume completed S1 selection as joint TTFT/decode32 final validation."""
import os,time,subprocess,signal,json
from pathlib import Path
from strict_headroom_common import *
from strict_headroom_guard import guard,cooldown
from run_strict_headroom_pipeline import publish
from batch_comm_common import stop_idle_load,start_idle_load
import run_timing_stability as h

def run_cell(row):
 world=row['world'];gpus=R4_GPUS if world==4 else list(range(8));label=f'S2D32_R{world}_B{row["batch"]}_L{row["context"]}'
 out=ROOT/'physical'/label;out.mkdir(exist_ok=False)
 path=ROOT/'inputs'/f'R{world}_B{row["batch"]}_L{row["context"]}_s{row["sample_seed"]}_d{row["dp_seed"]}_r{row["br_seed"]}'
 spec=dict(row,inputs=str(path),order=(row['br_seed']+row['dp_seed'])&1);write(out/'specs.json',[spec])
 state=dict(status='RUNNING',spec=spec,world=world,gpus=gpus,started_unix=time.time(),initial=cooldown(gpus),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip())
 paths=list((P/'mgo_v2').glob('*.py'))+[P/'examples/strict_decode32_worker.py',P/'examples/la_physical_worker.py',P/'scripts/env_offload_policy.py',P/'scripts/la_placement.py'];state['code']={str(p):sha(p) for p in paths}
 env=h.env_for('env1');env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),MGO_V2_PHYSICAL_GPUS=','.join(map(str,gpus)))
 cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'examples/strict_decode32_worker.py'),'--specs',str(out/'specs.json'),'--output',str(out)]
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
  state['finished_unix']=time.time();write(out/'status.json',state);receipt=PACKET/(label+'.json');write(receipt,state);publish('strict decode32: '+label+' '+state['status'],[receipt])
 return state

def main():
 state=dict(status='RUNNING',stage='WAIT_S1_HANDOFF',started_unix=time.time(),owner_amendment='Add32 decode steps and TPOT/E2E to final validation')
 def save():write(ROOT/'decode32_status.json',state)
 save()
 try:
  deadline=time.monotonic()+6*3600
  while not (ROOT/'decode32_handoff.json').exists():
   if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
   if time.monotonic()>deadline:raise TimeoutError('S1 handoff wait')
   time.sleep(2)
  # Allow the superseded supervisor to publish its cancellation and restore loads.
  deadline=time.monotonic()+180
  while Path('/proc/2360391/cmdline').exists() and Path('/proc/2360391/cmdline').read_bytes():
   if time.monotonic()>deadline:raise TimeoutError('old supervisor exit')
   time.sleep(1)
  s0=json.loads((PACKET/'STRICT_HEADROOM_S0.json').read_text());selected=[];selection=[]
  for world,batch,context in sorted({(x['world'],x['batch'],x['context']) for x in s0['rows']}):
   cell=[x for x in s0['rows'] if (x['world'],x['batch'],x['context'])==(world,batch,context)];values=[]
   for i,row in enumerate(cell):
    status=json.loads((ROOT/'physical'/f'S1_R{world}_B{batch}_L{context}_{i}'/'status.json').read_text());assert status['status']=='PASS';rr=status['result']['results'][0];values.append((rr['gain'],row))
   gain,row=max(values,key=lambda x:x[0]);selected.append(row);selection.append(dict(row=row,S1_selection_gain=gain))
  write(PACKET/'DECODE32_SELECTED.json',dict(status='PASS',selected=selection,selection_metric='S1 TTFT only; no decode-based reselection',handoff=json.loads((ROOT/'decode32_handoff.json').read_text())))
  publish('plan: freeze selected seeds for joint decode32 validation',[PACKET/'DECODE32_SELECTED.json'])
  state.update(stage='FINAL_DECODE32',selected=selected,completed=[]);save();stop_idle_load()
  for row in selected:
   result=run_cell(row);state['completed'].append(result['spec']);save()
  import report_strict_decode32
  report_strict_decode32.main();publish('results: strict TTFT TPOT E2E decode32',[PACKET/'DECODE32_RESULTS.json',PACKET/'DECODE32_RESULTS.md']);state['status']='PASS'
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();save();write(PACKET/'DECODE32_EXECUTION.json',state);write(ROOT/'resident_models.json',dict(processes=start_idle_load()));publish('strict decode32 pipeline '+state['status'],[PACKET/'DECODE32_EXECUTION.json'])
if __name__=='__main__':main()
