"""Owner-authorized transport matrix with checkpoints and conditional Env2."""
import json,os,signal,subprocess,time,statistics,hashlib
from pathlib import Path
import run_env_offload_cell as launcher
from validate_env_offload_plan import validate
from batch_comm_common import stop_idle_load
import run_timing_stability as harness
P=launcher.P
ROOT=Path('/home/hwlee/mgo-results/transport_stack_remeasure_20261004')
PACKET=P/'experiments/transport_stack_remeasure_20261004'
BRANCH='codex/coslot-comm-remeasure-20261004'
launcher.PACKET=PACKET
harness.ROOT=ROOT
write=launcher.write
state=dict(status='RUNNING',owner_commit='ac6d6bc90e88e9c9cd3a8791805124e0e57623e7',cell='R',physical_gpus=[0,1,4,5],stage='STARTING',started_unix=time.time(),plans={},completed=[])

def publish(message):
 subprocess.run(['git','add',str(PACKET)],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  subprocess.run(['git','push','origin','HEAD:'+BRANCH],cwd=P.parent,check=True)

def checkpoint():write(PACKET/'status.json',state)

def phase(policy,name,env,transport='current',h2d='pinned',repeat=0,plan=None):
 state.update(stage=name,environment=env,policy=policy,transport=transport,repeat=repeat);checkpoint()
 try:
  out=launcher.run('R',policy,name,env,repeat,transport,h2d,plan,2)
 except BaseException:
  # Also clean up if the lower-level launcher throws at a boundary assertion.
  for path in launcher.ROOT.glob(f'R_{policy}_{env}_{name}*/status.json'):
   receipt=json.loads(path.read_text());pid=receipt.get('pid');cmd=Path('/proc')/str(pid)/'cmdline'
   if receipt['status']=='RUNNING' and pid and cmd.exists() and b'env_offload_worker.py' in cmd.read_bytes() and os.getpgid(pid)==pid:
    os.killpg(pid,signal.SIGTERM)
    for _ in range(15):
     if not cmd.exists():break
     time.sleep(1)
    if cmd.exists():os.killpg(pid,signal.SIGKILL)
  raise
 state['completed'].append(str(out));checkpoint();publish(f'results: transport stack {env} {policy} {transport} {name} {repeat}')
 return out

def summarize(env):
 summaries=[];raw=[]
 for policy in ('BR','CA'):
  plan=Path(state['plans'][policy]);proof=json.loads((plan/'schedule_validation.json').read_text())
  for transport in ('current','coslot','coslot-active'):
   tag='pinned' if transport=='current' else transport+'_pinned';samples=[]
   for rep in range(3):
    out=launcher.ROOT/f'R_{policy}_{env}_MEASURE_{tag}_{rep}';rs=[json.loads((out/f'rank{r}.json').read_text()) for r in range(4)]
    assert all(x['status']=='PASS' and x['no_compile_in_measure'] for x in rs)
    for r,x in enumerate(rs):
     for k in ('token_hash','state_hash','action_hash','schedule_file_sha256'):assert x[k]==proof['ranks'][r][k]
    samples.append({k:max(x[k] for x in rs) for k in ('E2E_wall','decode_wall','TPOT')})
   out=launcher.ROOT/f'R_{policy}_{env}_COUNTERS_{tag}_0';rs=[json.loads((out/f'rank{r}.json').read_text()) for r in range(4)]
   for r,x in enumerate(rs):
    assert x['status']=='PASS' and x['actual_H2D_bytes']==proof['ranks'][r]['physical_H2D_bytes']
    assert x['actual_collective_calls']==(37008 if transport=='current' else 24672 if transport=='coslot' else 0)
    for k in ('token_hash','state_hash','action_hash','schedule_file_sha256'):assert x[k]==proof['ranks'][r][k]
    assert x['pinned_stage_bytes']==2*9437184
    if transport=='coslot-active':assert x['actual_p2p_batches']+x['actual_zero_remote_rounds']==24672
   metrics={k:dict(median=statistics.median(x[k] for x in samples),min=min(x[k] for x in samples),max=max(x[k] for x in samples)) for k in ('E2E_wall','decode_wall','TPOT')}
   for m in metrics.values():m['spread_relative']=(m['max']-m['min'])/m['median']
   counters={k:sum(x[k] for x in rs) for k in ('actual_H2D_bytes','actual_peer_bytes','actual_wire_bytes','actual_collective_calls','actual_p2p_batches','actual_p2p_ops','actual_zero_remote_rounds','active_peer_degree_sum')}
   counters['mean_active_peer_degree']=counters['active_peer_degree_sum']/(4*24672) if transport=='coslot-active' else None
   counters['max_active_peer_degree']=max(x['max_active_peer_degree'] for x in rs)
   summaries.append(dict(policy=policy,transport=transport,metrics=metrics,counters=counters,samples=samples))
 comparisons=[]
 for transport in ('current','coslot','coslot-active'):
  br,ca=[next(x for x in summaries if x['transport']==transport and x['policy']==p) for p in ('BR','CA')]
  comparisons.append(dict(transport=transport,**{k+'_CA_gain':1-ca['metrics'][k]['median']/br['metrics'][k]['median'] for k in ('E2E_wall','TPOT')},noise_overlap={k:abs(br['metrics'][k]['median']-ca['metrics'][k]['median'])<max(x['metrics'][k]['max']-x['metrics'][k]['min'] for x in (br,ca)) for k in ('E2E_wall','TPOT')}))
 stable=all(x['metrics'][k]['spread_relative']<=.05 for x in summaries for k in ('E2E_wall','TPOT'))
 for d in state['completed']:
  path=Path(d)
  if env in path.name:
   raw.extend(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in path.glob('rank*.json'))
 write(PACKET/(env+'_results.json'),dict(status='PASS',stable=stable,stability_gate='(max-min)/median <=5% for E2E and TPOT, all six conditions; no extra repeats',summaries=summaries,comparisons=comparisons,raw_receipts=raw))
 lines=[f'# {env} transport stack results','','Three measurements per condition. Raw medians and full ranges are in the JSON.','Stable timing does not imply a statistically established BR/CA difference.','','| Policy | Transport | TPOT median (s) | E2E median (s) | TPOT spread |','|---|---|---:|---:|---:|']
 for x in summaries:lines.append(f"| {x['policy']} | {x['transport']} | {x['metrics']['TPOT']['median']:.6f} | {x['metrics']['E2E_wall']['median']:.3f} | {x['metrics']['TPOT']['spread_relative']:.2%} |")
 lines+=['',f'Stability gate: {stable}. Env2 is conditional on all Env1 conditions passing.']
 (PACKET/(env+'_RESULTS.md')).write_text('\n'.join(lines)+'\n');publish('results: summarize transport stack '+env);return stable

def main():
 ROOT.mkdir(exist_ok=True);checkpoint()
 try:
  assert json.loads((PACKET/'CPU_validation.json').read_text())['status']=='PASS'
  stop_idle_load()
  stop=launcher.ROOT/'STOP'
  if stop.exists():stop.rename(ROOT/'superseded_previous_STOP')
  for policy in ('BR','CA'):
   plan=launcher.discover_validated_plan('R',policy)
   if plan is None:
    repeat=0
    while (launcher.ROOT/f'R_{policy}_env1_PLAN_{repeat}').exists():repeat+=1
    plan=phase(policy,'PLAN','env1','current','pageable',repeat)
    validate(plan);write(PACKET/(policy+'_plan_validation.json'),json.loads((plan/'schedule_validation.json').read_text()));publish('results: freeze reference R '+policy+' PLAN')
   state['plans'][policy]=str(plan);checkpoint()
  for env in ('env1','env2'):
   for policy in ('BR','CA'):
    for transport in ('current','coslot','coslot-active'):
     plan=Path(state['plans'][policy]);phase(policy,'COMPILE',env,transport,plan=plan)
     for repeat in range(3):phase(policy,'MEASURE',env,transport,repeat=repeat,plan=plan)
     phase(policy,'COUNTERS',env,transport,plan=plan)
   if not summarize(env):state.update(status='UNSTABLE_STOP',stage='FINISHED');break
  else:state.update(status='COMPLETE',stage='FINISHED')
 except BaseException as exc:state.update(status='FAILED_OR_STOPPED',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();checkpoint();state['resident_models']=harness.restore();checkpoint();publish('results: transport stack checkpoint and GPU handoff')
if __name__=='__main__':main()
