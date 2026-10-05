"""Independent packet accounting and balanced-quota tests for new policies."""
import numpy as np
from mgo_v2.fanout_admission import fanout_assignment,load_fanout_assignment,_prepare_packet_state,_candidate_packet_cost,_commit_packet_choice
rng=np.random.default_rng(17)
for seed in range(12):
 world=4;experts=128;n=24
 effective=rng.integers(0,12,(n,4));lengths=np.full(n,4,np.int64);origins=rng.integers(0,world,n)
 primary=np.full(48*experts,-1,np.int64);owner=np.zeros(48*experts,np.int64)
 for e in range(6):primary[e]=e%world;owner[e]=1<<(e%world)
 misses=np.arange(6,12);demand=np.zeros((experts,world),np.int64)
 for t in range(n):
  for e in set(effective[t]):demand[e,origins[t]]+=1
 a=fanout_assignment(effective,lengths,origins,primary,0,experts,misses,world)
 b=load_fanout_assignment(demand,effective,lengths,origins,owner,primary,0,experts,misses,world)
 for assigned in (a,b):
  assert np.array_equal(np.bincount(assigned,minlength=world),[2,2,1,1])
 presence,masks,incident=_prepare_packet_state(effective,lengths,origins,primary,0,experts,misses,world)
 def reference(placement):
  counts=np.zeros(world,np.int64)
  for t in range(n):
   for dst in {int(placement[e]) for e in effective[t] if placement[e]>=0}:
    if dst!=origins[t]:counts[dst]+=1;counts[origins[t]]+=1
  return counts
 current=primary.copy();assert np.array_equal(incident,reference(current))
 for i,e in enumerate(misses):
  r=int(a[i]);trial=current.copy();trial[e]=r;expect=reference(trial)
  mx,delta=_candidate_packet_cost(i,r,presence,masks,origins,incident,world)
  assert mx==expect.max() and delta==(expect.sum()-incident.sum())//2
  _commit_packet_choice(i,r,presence,masks,origins,incident);current=trial
  assert np.array_equal(incident,expect)
print('PASS: independent packet counts, incremental costs, balanced quotas, 12 fixtures')
