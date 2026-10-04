"""Owner-authorized c30/s1 transport matrix with checkpoints and conditional Env2."""
import hashlib,json,os,signal,statistics,subprocess,time,sys
from pathlib import Path
import run_env_offload_cell as launcher
from validate_env_offload_plan import validate
from batch_comm_common import stop_idle_load
import run_timing_stability as harness

P=launcher.P
ROOT=Path('/home/hwlee/mgo-results/transport_stack_remeasure_20261004/c30_s1')
PACKET=P/'experiments/transport_stack_remeasure_20261004/c30_s1'
BRANCH='codex/coslot-comm-remeasure-20261004'
CACHE_RATIO=0.30
SUBSTITUTION=True
CACHE_TAG='c30'
SUB_TAG='s1'
launcher.PACKET=PACKET
harness.ROOT=ROOT
write=launcher.write
state=dict(status='RUNNING',cell='R',cache_ratio=CACHE_RATIO,substitution=SUBSTITUTION,
 physical_gpus=[0,1,4,5],stage='STARTING',started_unix=time.time(),plans={},completed=[])

def publish(message):
 subprocess.run(['git','add',str(PACKET)],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  pushed=subprocess.run(['git','push','origin','HEAD:'+BRANCH],cwd=P.parent)
  if pushed.returncode:write(ROOT/'publication_pending.json',dict(status='LOCAL_COMMIT_SAVED_PUSH_PENDING',reason='remote advanced or push unavailable; do not interrupt scientific execution',unix=time.time()))

def checkpoint():write(PACKET/'status.json',state)

def run_cpu_validation():
 cmd=[sys.executable,str(P/'scripts/check_transport_cpu.py')]
 proc=subprocess.run(cmd,cwd=P,text=True,capture_output=True)
 receipt=dict(status='PASS' if proc.returncode==0 else 'FAIL',command=cmd,
  stdout=proc.stdout,stderr=proc.stderr,cache_ratio=CACHE_RATIO,substitution=SUBSTITUTION)
 write(PACKET/'CPU_validation.json',receipt)
 if proc.returncode:raise RuntimeError('CPU validation failed')
 publish('test: validate c30 s1 transport packet')

def phase(policy,name,env,transport='current',h2d='pinned',repeat=0,plan=None):
 state.update(stage=name,environment=env,policy=policy,transport=transport,repeat=repeat);checkpoint()
 try:
  out=launcher.run('R',policy,name,env,repeat,transport,h2d,plan,2,CACHE_RATIO,SUBSTITUTION)
 except BaseException:
  for path in launcher.ROOT.glob(f'R_{policy}_{env}_{name}_{CACHE_TAG}_{SUB_TAG}*/status.json'):
   receipt=json.loads(path.read_text());pid=receipt.get('pid');cmd=Path('/proc')/str(pid)/'cmdline'
   if receipt['status']=='RUNNING' and pid and cmd.exists() and b'env_offload_worker.py' in cmd.read_bytes() and os.getpgid(pid)==pid:
    os.killpg(pid,signal.SIGTERM)
    for _ in range(15):
     if not cmd.exists():break
     time.sleep(1)
    if cmd.exists():os.killpg(pid,signal.SIGKILL)
  raise
 state['completed'].append(str(out));checkpoint();publish(f'results: c30 s1 {env} {policy} {transport} {name} {repeat}')
 return out

def result_dir(policy,env,phase_name,transport,repeat):
 tags=[CACHE_TAG,SUB_TAG]
 if transport!='current':tags.append(transport)
 tags.append('pinned')
 return launcher.ROOT/f"R_{policy}_{env}_{phase_name}_{'_'.join(tags)}_{repeat}"

def summarize(env):
 summaries=[];raw=[]
 for policy in ('BR','CA'):
  plan=Path(state['plans'][policy]);proof=json.loads((plan/'schedule_validation.json').read_text())
  for transport in ('current','coslot','coslot-active'):
   samples=[]
   for rep in range(3):
    out=result_dir(policy,env,'MEASURE',transport,rep)
    rs=[json.loads((out/f'rank{r}.json').read_text()) for r in range(4)]
    assert all(x['status']=='PASS' and x['no_compile_in_measure'] for x in rs)
    assert all(abs(x['cache_ratio']-CACHE_RATIO)<1e-9 and x['substitution'] is SUBSTITUTION for x in rs)
    for rank,x in enumerate(rs):
     for key in ('token_hash','state_hash','action_hash','schedule_file_sha256'):
      assert x[key]==proof['ranks'][rank][key]
    samples.append({k:max(x[k] for x in rs) for k in ('E2E_wall','decode_wall','TPOT')})
   out=result_dir(policy,env,'COUNTERS',transport,0)
   rs=[json.loads((out/f'rank{r}.json').read_text()) for r in range(4)]
   for rank,x in enumerate(rs):
    assert x['status']=='PASS' and x['actual_H2D_bytes']==proof['ranks'][rank]['physical_H2D_bytes']
    assert abs(x['cache_ratio']-CACHE_RATIO)<1e-9 and x['substitution'] is SUBSTITUTION
    assert x['actual_collective_calls']==(37008 if transport=='current' else 24672 if transport=='coslot' else 0)
    for key in ('token_hash','state_hash','action_hash','schedule_file_sha256'):
     assert x[key]==proof['ranks'][rank][key]
    assert x['pinned_stage_bytes']==2*9437184
    if transport=='coslot-active':
     assert x['actual_p2p_batches']+x['actual_zero_remote_rounds']==24672
   metrics={k:dict(median=statistics.median(x[k] for x in samples),min=min(x[k] for x in samples),max=max(x[k] for x in samples)) for k in ('E2E_wall','decode_wall','TPOT')}
   for m in metrics.values():m['spread_relative']=(m['max']-m['min'])/m['median']
   counters={k:sum(x[k] for x in rs) for k in ('actual_H2D_bytes','actual_peer_bytes','actual_wire_bytes','actual_collective_calls','actual_p2p_batches','actual_p2p_ops','actual_zero_remote_rounds','active_peer_degree_sum')}
   counters['mean_active_peer_degree']=counters['active_peer_degree_sum']/(4*24672) if transport=='coslot-active' else None
   counters['max_active_peer_degree']=max(x['max_active_peer_degree'] for x in rs)
   summaries.append(dict(policy=policy,transport=transport,metrics=metrics,counters=counters,samples=samples))
 comparisons=[]
 for transport in ('current','coslot','coslot-active'):
  br,ca=[next(x for x in summaries if x['transport']==transport and x['policy']==policy) for policy in ('BR','CA')]
  comparisons.append(dict(transport=transport,
   **{k+'_CA_gain':1-ca['metrics'][k]['median']/br['metrics'][k]['median'] for k in ('E2E_wall','TPOT')},
   noise_overlap={k:abs(br['metrics'][k]['median']-ca['metrics'][k]['median'])<max(x['metrics'][k]['max']-x['metrics'][k]['min'] for x in (br,ca)) for k in ('E2E_wall','TPOT')}))
 stable=all(x['metrics'][k]['spread_relative']<=.05 for x in summaries for k in ('E2E_wall','TPOT'))
 for directory in state['completed']:
  path=Path(directory)
  if env in path.name:
   raw.extend(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in path.glob('rank*.json'))
 write(PACKET/(env+'_results.json'),dict(status='PASS',cache_ratio=CACHE_RATIO,substitution=SUBSTITUTION,stable=stable,
  stability_gate='(max-min)/median <=5% for E2E and TPOT, all six conditions; no extra repeats',
  summaries=summaries,comparisons=comparisons,raw_receipts=raw))
 lines=[f'# {env} c30/s1 transport stack results','',
  'Three measurements per condition. Raw medians and full ranges are in the JSON.',
  'Stable timing does not imply a statistically established BR/CA difference.','',
  '| Policy | Transport | TPOT median (s) | E2E median (s) | TPOT spread |',
  '|---|---|---:|---:|---:|']
 for x in summaries:lines.append(f"| {x['policy']} | {x['transport']} | {x['metrics']['TPOT']['median']:.6f} | {x['metrics']['E2E_wall']['median']:.3f} | {x['metrics']['TPOT']['spread_relative']:.2%} |")
 lines+=['',f'Stability gate: {stable}. Env2 is conditional on all Env1 conditions passing.']
 (PACKET/(env+'_RESULTS.md')).write_text('\n'.join(lines)+'\n')
 publish('results: summarize c30 s1 transport stack '+env);return stable

def main():
 ROOT.mkdir(parents=True,exist_ok=True);PACKET.mkdir(parents=True,exist_ok=True);checkpoint()
 try:
  run_cpu_validation()
  # Recover a successful reference generation whose publication previously
  # interrupted validation, instead of rerunning the GPU PLAN.
  for path in launcher.ROOT.glob('R_*_env1_PLAN_*'):
   status=path/'status.json';receipt=path/'rank0.json'
   if not status.exists() or not receipt.exists() or (path/'schedule_validation.json').exists():continue
   a=json.loads(status.read_text());b=json.loads(receipt.read_text());spec=b.get('cell_spec',{})
   if a.get('status')=='PASS' and abs(b.get('cache_ratio',spec.get('cache_ratio',-1))-CACHE_RATIO)<1e-9 and b.get('substitution',spec.get('substitution')) is SUBSTITUTION:
    validate(path)
  stop_idle_load()
  stop=launcher.ROOT/'STOP'
  if stop.exists():stop.rename(ROOT/'superseded_previous_STOP')
  for policy in ('BR','CA'):
   plan=launcher.discover_validated_plan('R',policy,CACHE_RATIO,SUBSTITUTION)
   if plan is None:
    repeat=0
    while (launcher.ROOT/f'R_{policy}_env1_PLAN_{CACHE_TAG}_{SUB_TAG}_{repeat}').exists():repeat+=1
    plan=phase(policy,'PLAN','env1','current','pageable',repeat)
    validate(plan)
    write(PACKET/(policy+'_c30_s1_plan_validation.json'),json.loads((plan/'schedule_validation.json').read_text()))
    publish('results: freeze c30 s1 reference R '+policy+' PLAN')
   state['plans'][policy]=str(plan);checkpoint()
  for env in ('env1','env2'):
   for policy in ('BR','CA'):
    for transport in ('current','coslot','coslot-active'):
     plan=Path(state['plans'][policy]);phase(policy,'COMPILE',env,transport,plan=plan)
     for repeat in range(3):phase(policy,'MEASURE',env,transport,repeat=repeat,plan=plan)
     phase(policy,'COUNTERS',env,transport,plan=plan)
   if not summarize(env):state.update(status='UNSTABLE_STOP',stage='FINISHED');break
  else:state.update(status='COMPLETE',stage='FINISHED')
 except BaseException as exc:
  state.update(status='FAILED_OR_STOPPED',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();checkpoint();state['resident_models']=harness.restore();checkpoint();publish('results: c30 s1 transport checkpoint and GPU handoff')

if __name__=='__main__':main()
