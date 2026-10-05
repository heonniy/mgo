"""Checkpointed queue; no process/GPU scans inside clean MEASURE regions."""
import os,json,subprocess,signal,time,argparse,re
from pathlib import Path
import run_timing_stability as h
from batch_comm_common import stop_idle_load,start_idle_load
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004');PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';h.ROOT=ROOT

def boundary_diagnostics():
 row=dict(load_average=os.getloadavg())
 try:
  text=subprocess.check_output(['nvidia-smi','--query-gpu=index,clocks.sm,clocks.mem,power.draw,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
  row['gpu_clock_power_utilization_csv']=text.strip().splitlines()
 except (subprocess.SubprocessError,OSError) as exc:row['diagnostic_error']=repr(exc)
 return row

def publish(message,paths):
 subprocess.run(['git','add',*[str(p) for p in paths]],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  if subprocess.run(['git','push','origin','HEAD:'+os.environ.get('MGO_RESULT_BRANCH','refactoring')],cwd=P.parent).returncode:h.write(ROOT/'publication_pending.json',dict(message=message,unix=time.time()))

def run(stage,group):
 world=group.get('world',8);gpus=group.get('gpus',list(range(world)));assert len(gpus)==world
 label=f'{stage}_B{group["batch"]}_H{group["horizon"]}';out=ROOT/label
 if (out/'status.json').exists():
  saved=json.loads((out/'status.json').read_text());assert saved['status']=='PASS','use a new attempt label after repair';return saved
 out.mkdir(exist_ok=True);h.write(out/'cases.json',group['cases']);receipt=PACKET/(label+'.json')
 state=dict(status='RUNNING',stage=stage,label=label,group=group,started_unix=time.time(),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parent,text=True).strip(),monitor='boundaries only; file polling during MEASURE')
 h.safe(h.sample(),True);env=h.env_for(group.get('environment','env1'));env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),MGO_V2_PHYSICAL_GPUS=','.join(map(str,gpus)));cache=ROOT/'compile_cache';env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
 if group.get('environment')=='env2':env['NCCL_DEBUG']='INFO'
 cmd=[h.PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/('examples/policy_regime_worker.py' if group.get('policy_regime') else ('examples/refactor_paired_worker.py' if group.get('paired') else 'examples/refactor_measure_worker.py'))),'--inputs',str(ROOT/f'inputs_B{group["batch"]}_H{group["horizon"]}'),'--cases',str(out/'cases.json'),'--output',str(out)]
 if group.get('paired'):
  from refactor_fingerprint import capture
  state['common_stack']=capture(ROOT/f'inputs_B{group["batch"]}_H{group["horizon"]}',env)
 active=None;released=set();last_scan=0
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;h.write(out/'status.json',state);h.write(PACKET/'status.json',state)
  try:
   while proc.poll() is None:
    if time.time()-state['started_unix']>8*3600:raise TimeoutError('bounded eight-hour batch group')
    if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
    if active is not None:
     files=list(out.glob(f'{active}_measure_rank*.json'))
     if len(files)==world:
      rows=[json.loads((out/f'{active}_measure_rank{r}.json').read_text()) for r in range(world)];record=PACKET/f'{label}_{active}.json';h.write(record,dict(status='PASS',source_sha=state['source_sha'],ranks=rows));publish(f'{stage}: {label} {active} clean physical sample',[record])
      state.setdefault('completed',[]).append(active);active=None;h.write(out/'status.json',state)
    else:
     boundary=json.loads((out/'boundary.json').read_text()) if (out/'boundary.json').exists() else None
     if boundary and boundary['key'] not in released:
      if group.get('environment')=='env2' and 'transport_verification' not in state:
       paths=set(re.findall(r'via (SHM|P2P|NET)/',(out/'run.log').read_text()))
       assert paths=={'SHM'},('Env2 requires verified SHM channels',paths)
       state['transport_verification']=dict(status='PASS',observed_paths=sorted(paths),scope='NCCL initialization channel log before first MEASURE; P2P and NET absent')
      key=boundary['key'];assert len(list(out.glob(f'{key}_ready_rank*.json')))==world
      sample=h.sample(proc.pid);h.safe(sample);state.setdefault('boundaries',[]).append(dict(key=key,sample=sample,diagnostics=boundary_diagnostics()));h.write(out/'status.json',state)
      h.write(PACKET/'status.json',dict(status='RUNNING',stage=stage,label=label,active=boundary,started_unix=state['started_unix'],pid=proc.pid))
      (out/f'{key}_GO').touch();released.add(key);active=key
     elif time.monotonic()-last_scan>30:
      h.safe(h.sample(proc.pid));last_scan=time.monotonic()
    time.sleep(2)
   assert proc.returncode==0,str(out/'run.log')
   result=json.loads((out/'result.json').read_text());assert result['status']=='PASS';state.update(status='PASS',result=result)
   # Process may exit just before the last file-poll iteration. Preserve that
   # sample as well, without relying on timing of parent/worker completion.
   for key in released:
    record=PACKET/f'{label}_{key}.json'
    if not record.exists():
     rows=[json.loads((out/f'{key}_measure_rank{r}.json').read_text()) for r in range(world)];h.write(record,dict(status='PASS',source_sha=state['source_sha'],ranks=rows));publish(f'{stage}: {label} {key} clean physical sample',[record])
  except BaseException as exc:
   state.update(status='FAIL',error=repr(exc))
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  finally:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   state.update(exit_code=proc.returncode,finished_unix=time.time());h.write(out/'status.json',state);h.write(receipt,state);h.write(PACKET/'status.json',{k:v for k,v in state.items() if k not in ['boundaries','result']});publish(f'{stage}: {label} {state["status"]}',[receipt,PACKET/'status.json'])
 return state

def main(a):
 global ROOT
 ROOT=a.root;ROOT.mkdir(parents=True,exist_ok=True);h.ROOT=ROOT
 assert subprocess.check_output(['git','branch','--show-current'],cwd=P.parent,text=True).strip()=='refactoring'
 groups=json.loads(a.config.read_text());stop_idle_load()
 try:
  for group in groups:
   deadline=time.monotonic()+180
   while any(g['temperature_c']>=65 for g in h.snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
   run(a.stage,group)
  h.write(PACKET/'status.json',dict(status='COMPLETE',stage=a.stage,finished_unix=time.time()))
 finally:
  if not (ROOT/'STOP').exists():h.write(ROOT/(a.stage+'_resident_models.json'),dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--stage',required=True);p.add_argument('--config',type=Path,required=True);main(p.parse_args())
