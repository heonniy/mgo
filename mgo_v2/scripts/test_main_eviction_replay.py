"""Analytical history semantics and exact BR/Gate replay tests."""
import numpy as np
from main_eviction_replay import replay
import mgo_v2
from env_offload_policy import Policy

counts=np.zeros((7,4),np.int32)
for i,active in enumerate(([0],[0],[0],[1],[1,2],[0],[3])):counts[i,active]=1
gates=np.zeros_like(counts,dtype=np.float32);cap=np.array([2],np.int32)
r=[replay(counts,gates,cap,m,layers=1) for m in range(5)]
assert set(r[1][1].ravel())=={1,3}
assert set(r[2][1].ravel())=={0,3}
assert r[1][2][0]==0 and r[2][2][0]==4
assert np.array_equal(r[3][0],r[4][0]) and np.array_equal(r[3][1],r[4][1])
assert all(np.all(x[0][:,3]<=x[0][:,1]) for x in r)
# Differential test against production policy: same gate eviction and BR RNG.
rng=np.random.default_rng(123)
for case in range(12):
 events=96;counts=np.zeros((events,128),np.int32);gates=rng.random((events,128)).astype(np.float32);cap=np.array([40,40,40,40],np.int32)
 for i in range(events):counts[i,rng.choice(128,12,replace=False)]=1
 actual,slots,_,_=replay(counts,gates,cap,0)
 policy=Policy(cap,np.zeros((48,128,128),np.float32),False,0,42);expected=[];seen=set()
 for i in range(events):
  sel=np.flatnonzero(counts[i]).reshape(-1,1);keys=i%48*128+sel.ravel();miss=[int(k) for k in keys if policy.owner[k]==0]
  out=policy.apply(i,sel,np.ones_like(sel,dtype=np.float32),np.arange(len(sel),dtype=np.int64)%4,gates[i],np.zeros((128,4),np.int32));fetches=out[5]
  expected.append([len(keys)-len(miss),len(miss),sum(f[3]>=0 for f in fetches),sum(k in seen for k in miss)]);seen.update(miss)
 assert np.array_equal(actual,expected) and np.array_equal(slots,policy.slots),case
print('PASS: LFU reset/cumulative divergence; reset deletes metadata; LRU equivalence; 12 production Gate/BR differential cases')
