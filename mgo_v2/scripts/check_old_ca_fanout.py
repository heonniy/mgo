"""CPU differential tests against the exact historical source, not a rewrite."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import ast, hashlib, json, subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linear_sum_assignment
from old_ca_fanout_policy import incremental_costs, fanout_assignment
from env_offload_policy import Policy
from fetch_matched_replay import replay_critical
from br_carep_cpu import replay

P=Path(__file__).resolve().parents[1]
COMMIT='a0be82e9000d7d7634db04c9558d64f8f896ee6d'
source=subprocess.check_output(['git','show',COMMIT+':mgo_v2/mgo_v2/admission.py'],cwd=P.parent,text=True)
keep={'balanced_quotas','_token_sources','_base_destinations','_incremental_cost','AdmissionPolicy','HungarianAdmission'}
tree=ast.parse(source);tree.body=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in keep]
ns=dict(np=np,linear_sum_assignment=linear_sum_assignment,AdmissionContext=object,AdmissionResult=lambda a,q,n:SimpleNamespace(expert_to_rank=a,quotas=q,name=n))
exec(compile(tree,'historical_admission.py','exec'),ns)
rng=np.random.default_rng(20261004);checks=0
for world in (1,4,8):
 for trial in range(40):
  experts=16;n=23;routes=np.array([rng.choice(experts,4,replace=False) for _ in range(n)],np.int16);lengths=np.full(n,4,np.int8);orig=rng.integers(world,size=n,dtype=np.int8)
  primary=np.full(experts,-1,np.int8)
  resident=rng.choice(experts,trial%17,replace=False);primary[resident]=rng.integers(world,size=len(resident))
  incoming=np.array(sorted(set(routes.ravel())-set(resident)),np.int64)
  ctx=SimpleNamespace(incoming=incoming.tolist(),origin_ranks=orig,effective_token_routes=[dict.fromkeys(x.tolist(),.25) for x in routes],preowned={int(e):int(primary[e]) for e in resident},world_size=world,affinity=None)
  base=ns['_base_destinations'](ctx);before=[set(x) for x in base];sources=ns['_token_sources'](ctx)
  expected=np.array([[ns['_incremental_cost'](ctx,int(e),r,sources,base) for r in range(world)] for e in incoming]).reshape(len(incoming),world)
  actual=incremental_costs(routes,lengths,orig,primary,0,experts,incoming,world)
  assert np.array_equal(expected,actual)
  ref=ns['HungarianAdmission']().place(ctx)
  saved=primary.copy();assigned=fanout_assignment(routes,lengths,orig,primary,0,experts,incoming,world)
  assert dict(zip(incoming.tolist(),assigned.tolist()))==ref.expert_to_rank
  assert np.array_equal(primary,saved) and base==before
  assert np.array_equal(np.bincount(assigned,minlength=world),ref.quotas)
  checks+=1
# Explicit coalescing: both misses share a pre-owned rank-1 destination;
# neither incoming expert is allowed to create a base destination for the other.
routes=np.array([[0,1,2]],np.int16);lengths=np.array([3],np.int8);orig=np.array([0],np.int8);primary=np.array([1,-1,-1],np.int8);incoming=np.array([1,2],np.int64)
assert incremental_costs(routes,lengths,orig,primary,0,3,incoming,4).tolist()==[[0,0,1,1],[0,0,1,1]]
# Exercise cache eviction and the complete incremental PLAN adapter against
# the compiled batch replay; BR/CA must also match the unchanged baseline.
for world in (4,8):
 layers=3;experts=16;horizon=5;n=world*2;events=layers*(horizon+1)
 selected=np.array([rng.choice(experts,3,replace=False) for _ in range(n*events)],np.uint8)
 weights=np.full(selected.shape,1/3,np.float32);offsets=np.arange(events+1,dtype=np.int64)*n
 origins=np.repeat(np.arange(world,dtype=np.int8),2);gates=rng.random((events,experts),dtype=np.float32)
 sim=np.zeros((layers,experts,experts),np.float32);cap=np.full(world,6,np.int32)
 args=(selected,weights,offsets,origins,origins,gates,sim,cap,horizon,True,False)
 for policy in (0,1,3):
  batch=replay_critical(*args,policy,42)
  if policy!=3:
   ref=replay(*args,policy,42)
   assert all(np.array_equal(a,b) for a,b in zip(batch[:-1],ref))
  adapter=Policy(cap,sim,False,policy,42);rows=[]
  for event in range(events):
   lo,hi=offsets[event:event+2]
   result=adapter.apply(event,selected[lo:hi],weights[lo:hi],origins,gates[event],np.zeros((experts,world),np.int32));rows.append(result[-1])
  assert np.array_equal(rows,batch[0])
  for name,expected in zip(('slots','owner','primary','last','seen','lost','birth','reuses'),batch[2:10]):assert np.array_equal(getattr(adapter,name),expected),name
receipt=dict(status='PASS',historical_commit=COMMIT,historical_source_sha256=hashlib.sha256(source.encode()).hexdigest(),randomized_historical_cost_and_assignment_cases=checks,preowned_coalescing=True,immutable_hungarian_base=True,balanced_quotas=True,BR_CA_unchanged_baseline_parity=True,incremental_vs_batch_cache_trajectory_parity=True,new_trace_capture=False)
(P/'experiments/old_ca_fanout_followup_20261004/policy_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
