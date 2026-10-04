"""Owner rule: two clean samples; at most one conditional third sample."""
import math,itertools
KEYS=('E2E_wall','TPOT')
def difference(a,b):
 assert math.isfinite(a) and math.isfinite(b) and a>0 and b>0
 return abs(a-b)/((a+b)/2)
def decide(rows):
 assert len(rows)==2,'Exactly two clean samples are required for the gate'
 delta={k:difference(rows[0][k],rows[1][k]) for k in KEYS};worst=max(delta.values())
 if worst>0.05+1e-12:status='UNSTABLE';target=2
 elif worst>0.02+1e-12:status='THIRD_REQUIRED';target=3
 else:status='STABLE_TWO';target=2
 return dict(status=status,target_repeats=target,relative_difference=delta,estimator='median' if target==3 else 'mean_and_median',rule='abs(T1-T2)/mean(T1,T2); E2E and TPOT only')
def final_unstable(rows):
 # Descriptive flag only: an outlying third sample cannot trigger another run.
 return any(difference(a[k],b[k])>0.05+1e-12 for a,b in itertools.combinations(rows,2) for k in KEYS)
def run(phase,samples,write,packet,commit):
 orders=[[('env1','BR'),('env1','CA'),('env2','CA'),('env2','BR')],[('env2','BR'),('env2','CA'),('env1','CA'),('env1','BR')]]
 for repeat,order in enumerate(orders,1):
  for env,policy in order:phase(policy,env,'MEASURE',repeat)
 gates=[]
 for env in ['env1','env2']:
  for policy in ['BR','CA']:
   rows=samples(policy,env);first=[r for r in rows if r['repeat'] in [1,2]]
   decision=decide(first);assert len(rows)<=decision['target_repeats']
   gates.append(dict(environment=env,policy=policy,**decision))
 write(packet/'noise_gate.json',gates);commit('results: owner two-sample E2E TPOT repetition decisions')
 for env,policy in [('env1','CA'),('env1','BR'),('env2','BR'),('env2','CA')]:
  gate=next(g for g in gates if (g['environment'],g['policy'])==(env,policy))
  if gate['target_repeats']==3:phase(policy,env,'MEASURE',3)
