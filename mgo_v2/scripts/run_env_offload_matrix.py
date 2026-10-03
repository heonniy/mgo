"""Bounded physical matrix after first PLAN validation; checkpoint every phase."""
import json,statistics,subprocess,time
from pathlib import Path
from run_env_offload_cell import run,ROOT,PACKET,P
from validate_env_offload_plan import validate
CELLS={'P':['BR','CA','CA-rep'],'R':['BR','CA','CA-rep'],'E':['BR','CA']}
ENVS={'P':['env1','env2'],'R':['env2','env1'],'E':['env1','env2']}
def commit(message):
 subprocess.run(['git','add',str(PACKET)],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  subprocess.run(['git','push','origin','HEAD:codex/mgo-r4-trajectory-results-20261002'],cwd=P.parent,check=True)
def phase(cell,policy,name,env,repeat=0):
 if name=='PLAN':
  candidates=[p for p in ROOT.glob(f'{cell}_{policy}_{env}_PLAN_*') if (p/'status.json').exists() and json.loads((p/'status.json').read_text())['status']=='PASS']
 else:candidates=[ROOT/f'{cell}_{policy}_{env}_{name}_{repeat}']
 if candidates:
  assert len(candidates)==1
  path=candidates[0]
  if (path/'status.json').exists():
   assert json.loads((path/'status.json').read_text())['status']=='PASS',str(path)
   if name=='PLAN' and not (path/'schedule_validation.json').exists():
    proof=validate(path);(PACKET/f'{cell}_{policy}_plan_validation.json').write_text(json.dumps(proof,indent=2)+'\n');commit(f'results: validate {cell} {policy} frozen PLAN')
   return path
 out=run(cell,policy,name,env,repeat)
 if name=='PLAN':
  proof=validate(out);(PACKET/f'{cell}_{policy}_plan_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
 commit(f'results: checkpoint {cell} {policy} {env} {name} {repeat}');return out
def samples(cell,policy,env):
 result=[]
 for path in ROOT.glob(f'{cell}_{policy}_{env}_MEASURE_*'):
  if json.loads((path/'status.json').read_text())['status']!='PASS':continue
  rows=[json.loads(p.read_text()) for p in path.glob('rank*.json')]
  result.append(max(r['E2E_wall'] for r in rows))
 return result
def main():
 # Validate first cell end to end before preparing the rest of the matrix.
 phase('P','BR','PLAN','env1');phase('P','BR','COMPILE','env1')
 for cell,policies in CELLS.items():
  for policy in policies:phase(cell,policy,'PLAN','env1')
  for env in ENVS[cell]:
   for policy in policies:phase(cell,policy,'COMPILE',env)
  orders=[['BR','CA','CA-rep'],['CA-rep','CA','BR'],['CA','BR','CA-rep']] if cell!='E' else [['BR','CA'],['CA','BR'],['BR','CA']]
  for rep,order in enumerate(orders,1):
   for env in ENVS[cell]:
    for policy in order:phase(cell,policy,'MEASURE',env,rep)
  for env in ENVS[cell]:
   for policy in policies:phase(cell,policy,'COUNTERS',env)
  decisions=[]
  for env in ENVS[cell]:
   extras=set()
   pairs=[('BR','CA')] if cell=='E' else [('BR','CA'),('CA','CA-rep')] if cell=='P' else [('CA','CA-rep')]
   for left,right in pairs:
    a=samples(cell,left,env);b=samples(cell,right,env);assert len(a)==len(b)==3
    delta=abs(statistics.median(a)-statistics.median(b));width=max(max(a)-min(a),max(b)-min(b));trigger=delta<width
    decisions.append(dict(environment=env,left=left,right=right,median_delta=delta,max_within_policy_range=width,extra_repeats=2 if trigger else 0))
    if trigger:extras.update([left,right])
   (PACKET/f'noise_gate_{cell}.json').write_text(json.dumps(decisions,indent=2)+'\n')
   commit(f'results: record {cell} {env} targeted noise decision')
   for rep in (4,5):
    for policy in ([p for p in reversed(policies) if p in extras] if rep==4 else [p for p in policies if p in extras]):phase(cell,policy,'MEASURE',env,rep)
 print('BOUNDED_MATRIX_COMPLETE',flush=True)
if __name__=='__main__':main()
