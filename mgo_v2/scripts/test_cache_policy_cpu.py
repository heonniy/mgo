"""Differential policy checks, including duplicate coverage and ragged routing."""
import json
from pathlib import Path
import numpy as np
from cache_policy_cpu import CachePolicyReplay,midrank2
from replica_pareto_cpu import ReplicaReplay,traffic
from mgo_v2.eviction import _midrank2
from mgo_v2.substitution import SubstitutionPolicy,merge_effective_routes
from mgo_v2.types import LayerRoutes

def main():
 rng=np.random.default_rng(17);checks=[]
 for rho in (0,.125,.25,.5,.75):
  a=ReplicaReplay([13]*4,rho);b=CachePolicyReplay([13]*4,rho,'lru',np.ones((5,10,10),dtype=np.float32))
  for tick in range(60):
   layer=tick%5;origins=np.repeat(np.arange(4),3);selected=np.array([rng.choice(10,3,False) for _ in origins])
   x=a.event(layer,origins,selected);y=b.event(layer,origins,selected,audit_greedy=tick<3)
   assert x[0]==y[0] and a.state()==b.state() and x[3]==y[3]
  checks.append(f'LRU rho={rho}: 60 action/counter/state matches')
 for size in (1,2,17,600):
  values=rng.integers(0,9,size);ref=_midrank2({(0,i):int(x) for i,x in enumerate(values)})
  assert midrank2(values).tolist()==[ref[0,i] for i in range(size)]
 checks.append('integer tied percentile ranks match production')
 sim=rng.random((5,10,10),dtype=np.float32)
 for eviction in ('gate','coverage'):
  b=CachePolicyReplay([13]*4,.75,eviction,sim)
  original=b.slot_or_victim
  def checked(rank,active):
   choice=original(rank,active)
   if choice and choice[1] is not None:
    legal=[k for k in b.ranks[rank] if k not in active]
    if eviction=='gate':score={k:float(b.gates[k]) for k in legal}
    else:
     damage={}
     for key in legal:
      l,e=key
      if len(b.owners[key])>1:damage[key]=0;continue
      residents=[x for ll,x in b.owners if ll==l]
      counts=b.neighbors[l][:,residents].sum(axis=1)
      damage[key]=int(b.neighbors[l,:,e][counts<=1].sum())
     gr=_midrank2({k:float(b.gates[k]) for k in legal});dr=_midrank2(damage)
     score={k:gr[k]+2*dr[k] for k in legal}
    ref=min(legal,key=lambda k:(score[k],b.ranks[rank][k].used,k))
    assert choice[1]==ref,(eviction,choice[1],ref)
   return choice
  b.slot_or_victim=checked
  for tick in range(35):
   b.gates[tick%5]=rng.integers(0,5,10)
   origins=np.repeat(np.arange(4),3);selected=[rng.choice(10,int(rng.integers(1,4)),False).tolist() for _ in origins]
   b.event(tick%5,origins,selected,audit_greedy=tick<3)
  checks.append(eviction+': 35 ragged events; every victim versus independent global-unique calculation')
 # Production substitution: any high source weight protects all its routes,
 # tier1 precedes inactive residents; duplicate targets merge per token.
 sim=np.ones((1,6,6),dtype=np.float32)*.7
 b=CachePolicyReplay([4]*4,0,'lru',sim,True)
 b.begin_event(set());b.place(0,(0,0),(0,None));b.place(1,(0,5),(0,None))
 r=LayerRoutes(0,np.array([0,1]),np.array([[0,1,2],[1,2,3]]),np.array([[.1,.21,.1],[.01,.02,.1]],dtype=np.float32))
 d=SubstitutionPolicy(sim).decide(r,b);assert d.protected_misses=={1} and d.source_to_target=={2:0,3:0}
 effective=merge_effective_routes(r,d);assert list(effective[0])==[0,1] and list(effective[1])==[1,0]
 row,ds,t,ops=b.event(0,r.origin_ranks,[list(x) for x in effective],audit_greedy=True)
 assert row['raw_expert_routes']==4 and row['total_fetches']==1
 checks.append('protected-source, tier1, merged effective routes and mandatory exact admission')
 p=Path(__file__).resolve().parents[1]/'experiments/cache_eviction_substitution_20261003/policy_tests.json'
 p.write_text(json.dumps(dict(status='PASS',checks=checks),indent=2)+'\n');print(p.read_text())
if __name__=='__main__':main()
