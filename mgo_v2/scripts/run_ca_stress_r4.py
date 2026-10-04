"""Bounded R4 physical validation; checkpoint phases and restore owned model workers."""
import os,json,subprocess,signal,time,statistics,csv,hashlib,gzip,pickle
from pathlib import Path
import run_timing_stability as harness
from validate_env_offload_plan import validate
VARIANT=os.environ['MGO_R4_GPU_SET'];assert VARIANT in ('0123','0146');GPUS=[int(x) for x in VARIANT]
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/ca_stress_physical_validation_20261004')/('R4_'+VARIANT);PACKET=P/'experiments/ca_stress_physical_validation_20261004'/('R4_'+VARIANT)
PYTHON=harness.PYTHON;write=harness.write
harness.ROOT=ROOT;harness.PACKET=PACKET

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def commit(message):harness.commit(message)
def env_for(environment,policy='BR'):
 env=harness.env_for(environment);cache=ROOT/'compile_cache'/policy;cache.mkdir(parents=True,exist_ok=True)
 env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,GPUS)),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'));return env

def prepare():
 ROOT.mkdir(exist_ok=True);PACKET.mkdir(exist_ok=True);(PACKET/'receipts').mkdir(exist_ok=True)
 selected=P/'experiments/ca_stress_workload_search_20261004/selected_manifests/ShareGPT_R4_B8_c30.json';m=json.loads(selected.read_text())
 assert (m['sample_seed'],m['dp_seed'],m['world'],m['batch'],m['cache_percent'])==(205,4,4,8,30)
 source=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/ShareGPT_requests.json');assert sha(source)==m['pool_manifest_sha256'];requests=json.loads(source.read_text())['requests']
 assert len(set(m['request_ids']))==32 and sum(m['ranks'],[])==m['request_ids']
 obj=dict(selected_manifest_sha256=sha(selected),ranks=[[requests[i] for i in ids] for ids in m['ranks']]);write(ROOT/'requests.json',obj)
 write(PACKET/'frozen_workload.json',dict(selected_manifest=m,request_manifest_sha256=sha(ROOT/'requests.json'),physical_request_order='exact selected manifest rank order',seed=73,physical_gpus=GPUS,substitution=False,cache_slots=1843,decode=256,source_hashes={f:sha(P/f) for f in ['examples/ca_stress_r4_worker.py','examples/env_offload_worker.py','examples/timing_stability_residency_worker.py','scripts/env_offload_policy.py','scripts/env_offload_tensors.py','scripts/env_offload_layout.py']}))
 assert json.loads((P/'experiments/timing_stability_numa_20261004/status.json').read_text())['status']=='HARNESS_STABLE'
 commit('experiment: freeze one physical CA stress workload and stable harness')

def launch(label,cmd,env,measure=False):
 out=ROOT/label
 if (out/'status.json').exists():
  state=json.loads((out/'status.json').read_text());assert state['status']=='PASS',str(out);return out
 out.mkdir(exist_ok=False);initial=harness.sample();harness.safe(initial,True)
 state=dict(status='RUNNING',label=label,command=cmd,started_unix=time.time(),initial=initial,monitor='BOUNDARY' if measure else 'untimed safety polling')
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(out/'status.json',state)
  try:
   while proc.poll() is None:
    if measure and len(list(out.glob('ready*.json')))==4:
     boundary=harness.sample(proc.pid,True);harness.safe(boundary);state['before_measure']=boundary;state['GO_unix']=time.time();write(out/'status.json',state);(out/'GO').touch()
     proc.wait(timeout=3600);break
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
    if proc.poll() is not None:break
    if time.time()-state['started_unix']>7200:raise TimeoutError('bounded phase exceeded two hours')
    sample=harness.sample(proc.pid,False);harness.safe(sample);state['last_safety']=sample;write(out/'status.json',state)
   assert proc.returncode==0,str(out/'run.log')
   if measure:assert 'GO_unix' in state
   state['status']='PASS'
  except BaseException as exc:
   state.update(status='FAIL',error=repr(exc))
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  finally:
   state.update(exit_code=proc.returncode,finished_unix=time.time());write(out/'status.json',state);write(PACKET/'receipts'/(label+'.json'),state);commit('results: R4 '+VARIANT+' physical CA stress '+label+' '+state['status'])
 harness.safe(harness.sample());return out

def command(worker,out):return [PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(P/'examples'/worker),'--output',str(out)]
def preflight(environment):
 label='transport_'+environment;out=ROOT/label;env=env_for(environment);env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(out/'nccl-%h-%p.log'))
 launch(label,command('env_offload_preflight.py',out),env)
 lines=[line for p in out.glob('nccl-*.log') for line in p.read_text().splitlines() if 'via ' in line];paths=sorted({line.split('via ',1)[1].split()[0] for line in lines})
 assert paths==(['P2P/IPC'] if environment=='env1' else ['SHM/direct/direct']),paths
 write(PACKET/(label+'.json'),dict(status='PASS',paths=paths,raw_root=str(out)));(PACKET/(label+'.log')).write_text('\n'.join(lines)+'\n');commit('results: R4 '+VARIANT+' physical CA stress '+label+' verified')

def phase(policy,environment,kind,repeat=0):
 label=f'{policy}_{environment}_{kind}_{repeat}';out=ROOT/label;plan=ROOT/f'{policy}_env1_PLAN_0'
 state=json.loads((PACKET/'status.json').read_text());state.update(stage=label,updated_unix=time.time());write(PACKET/'status.json',state)
 cmd=command('ca_stress_r4_worker.py',out)+['--policy',policy,'--environment',environment,'--phase',kind]
 if kind!='PLAN':cmd+=['--plan',str(plan)]
 launch(label,cmd,env_for(environment,policy),kind=='MEASURE')
 if kind=='PLAN':
  proof=validate(out);write(PACKET/(policy+'_plan_validation.json'),proof);commit('results: audit physical CA stress '+policy+' frozen schedule')
 return out

def audit_native_trace():
 import numpy as np
 source=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/pool/ShareGPT')
 arrays={k:np.load(source/(k+'.npy'),mmap_mode='r') for k in ['decode_selected','prefill_selected','offsets','generated_tokens']}
 request=json.loads((ROOT/'requests.json').read_text());rows=[]
 for policy in ['BR','CA']:
  plan=ROOT/f'{policy}_env1_PLAN_0'
  for rank in range(4):
   ids=[r['request_id'] for r in request['ranks'][rank]]
   with gzip.open(plan/f'rank{rank}.pkl.gz','rb') as f:events=pickle.load(f)
   changed=0;total=0
   for i,e in enumerate(events):
    layer=i%48
    if i<48:
     expected=np.concatenate([arrays['prefill_selected'][layer,arrays['offsets'][req]:arrays['offsets'][req+1]] for req in ids])
    else:expected=arrays['decode_selected'][i//48-1,layer,ids]
    actual=e['selected'];assert actual.shape==expected.shape
    changed+=int(np.count_nonzero(actual!=expected));total+=actual.size
   tokens=np.load(plan/f'rank{rank}_tokens.npy');native=arrays['generated_tokens'][ids]
   rows.append(dict(policy=policy,rank=rank,selected_expert_entries=total,different_selected_entries=changed,different_token_entries=int(np.count_nonzero(tokens!=native)),total_tokens=int(tokens.size)))
   del events
 write(PACKET/'PLAN_CPU_comparison.json',dict(status='COMPARED',native_capture_is_not_a_physical_timing_run=True,rows=rows))
 commit('results: compare physical stress PLAN routes with native CPU reference')

def samples(policy,environment):
 rows=[]
 for p in ROOT.glob(f'{policy}_{environment}_MEASURE_*/status.json'):
  s=json.loads(p.read_text());assert s['status']=='PASS';r=[json.loads((p.parent/f'rank{i}.json').read_text()) for i in range(4)];assert all(a['status']=='PASS' and a['no_compile_in_measure'] for a in r)
  rows.append(dict(policy=policy,environment=environment,repeat=int(p.parent.name.rsplit('_',1)[1]),**{k:max(a[k] for a in r) for k in ['E2E_wall','decode_wall','TPOT']}))
 return sorted(rows,key=lambda r:r['repeat'])
def main():
 ROOT.mkdir(exist_ok=True);PACKET.mkdir(exist_ok=True);write(PACKET/'status.json',dict(status='RUNNING',stage='PREPARE',owner_commit='a121c85dc6820a36b03854a9c51e2109f89aca0c',physical_gpus=GPUS,started_unix=time.time()))
 try:
  prepare()
  subprocess.run([PYTHON,str(P/'scripts/check_ca_stress_r4.py')],env=env_for('env1'),check=True)
  assert json.loads((PACKET/'implementation_validation.json').read_text())['status']=='PASS'
  commit('results: validate R4 seed73 full256 incremental policy')
  from batch_comm_common import stop_idle_load
  stop_idle_load();deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in harness.snapshot()['gpus']) and time.monotonic()<deadline:time.sleep(5)
  for env in ['env1','env2']:preflight(env)
  for policy in ['BR','CA']:phase(policy,'env1','PLAN')
  audit_native_trace()
  for env in ['env1','env2']:
   for policy in ['BR','CA']:phase(policy,env,'COMPILE')
  from physical_repeat_rule import run as adaptive_measure
  adaptive_measure(phase,samples,write,PACKET,commit)
  for env in ['env1','env2']:
   for policy in ['BR','CA']:phase(policy,env,'COUNTERS')
  subprocess.run([PYTHON,str(P/'scripts/summarize_ca_stress_r4.py')],env=env_for('env1'),check=True)
  state=json.loads((PACKET/'status.json').read_text());state.update(status='COMPLETE',stage='FINISHED',finished_unix=time.time());write(PACKET/'status.json',state)
 except BaseException as exc:
  state=json.loads((PACKET/'status.json').read_text());state.update(status='FAILED_OR_STOPPED',error=repr(exc),finished_unix=time.time());write(PACKET/'status.json',state);raise
 finally:
  state=json.loads((PACKET/'status.json').read_text());state['resident_models']='DEFERRED_TO_R4_QUEUE' if os.environ.get('MGO_R4_CHAIN')=='1' else harness.restore();write(PACKET/'status.json',state);commit('results: R4 '+VARIANT+' physical CA stress checkpoint and GPU handoff')
if __name__=='__main__':main()
