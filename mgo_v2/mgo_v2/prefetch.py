"""Deterministic CPU prefetch shadow; physical slot roles integrate at M4.

The existing Policy owns MAIN arrays. This object owns only reservations and
uses the same MAIN arrays for promotions and subsequent demand admission.
"""
import numpy as np
from env_offload_policy import Policy
from br_carep_cpu import choose_slot,place,balanced_assignment
from la_placement import load_assignment
from .predictor import choose_candidates

class PrefetchShadow:
 def __init__(self,capacities,budget,policy,seed,predictor):
  self.policy_name=policy;self.seed=seed;self.budget=budget;self.predictor=predictor
  self.main=Policy(capacities,np.zeros((48,128,128),np.float32),False,{'BR':0,'CA':1,'LA':4}[policy],seed)
  self.pending={};self.counters=dict(issued=0,useful=0,wasted=0,promotions=0,promotion_evictions=0,promotion_victim_reloads=0,mandatory=0,quota_violations=0)
  self.promotion_victims=set();self.event=-1
 def plan_current(self,event,selected,weights,origins,gates):
  self.event=event;layer=event%48;tick=event+1;active=np.zeros(128,np.bool_);active[np.unique(selected)]=True
  m=self.main;m.gates[layer]=gates;promotions=[];discard=[]
  for key,(rank,pfslot,target) in sorted(self.pending.items()):
   assert target==layer and event>=48
   if active[key%128]:
    assert m.owner[key]==0
    slot=choose_slot(rank,layer,active,m.slots,m.capacities,m.last,m.gates,True,128);assert slot>=0
    victim=int(m.slots[rank,slot]);row=np.zeros(48,np.float64)
    place(rank,key,slot,False,tick,m.slots,m.owner,m.primary,m.last,m.seen,m.lost,m.birth,m.reuses,row)
    promotions.append((rank,key,slot,pfslot,victim));self.counters['useful']+=1;self.counters['promotions']+=1
    if victim>=0:self.promotion_victims.add(victim);self.counters['promotion_evictions']+=1
   else:discard.append((rank,key,pfslot));self.counters['wasted']+=1
  self.pending.clear()
  out=m.apply(event,selected,weights,origins,gates,np.zeros((128,8),np.int32));fetches=out[5]
  quotas=np.bincount([f[0] for f in fetches],minlength=8);assert quotas.max()-quotas.min()<=1
  for rank,key,slot,victim,rep in fetches:
   assert not rep
   if key in self.promotion_victims:self.counters['promotion_victim_reloads']+=1;self.promotion_victims.remove(key)
  self.counters['mandatory']+=len(fetches)
  return out,promotions,discard
 def plan_prefetch_next(self,per_rank_counts):
  layer=self.event%48
  if self.event<48 or layer==47 or self.budget==0:return []
  predicted=self.predictor.predict_next(layer,per_rank_counts)
  candidates=choose_candidates(predicted,self.main.slots.ravel(),self.pending.keys(),layer+1,self.budget)
  n=len(candidates)
  if not n:return []
  # Fixed-point scores preserve fractional predictor ordering for integer CA/LA.
  demand=np.rint(predicted.T*1_000_000).astype(np.int64)
  if self.policy_name=='BR':
   slots=np.repeat(np.arange(8),[n//8+(r<n%8) for r in range(8)]);order=np.random.default_rng(np.random.SeedSequence([self.seed,self.event,991])).permutation(n);assignment=np.empty(n,np.int64);assignment[order]=slots
  elif self.policy_name=='CA':assignment=balanced_assignment(demand,candidates,8,False)
  else:assignment=load_assignment(demand,candidates,self.main.owner,layer+1)
  counts=np.zeros(8,np.int64);result=[]
  for expert,rank in zip(candidates,assignment):
   rank=int(rank);key=(layer+1)*128+int(expert);assert self.main.owner[key]==0 and key not in self.pending
   slot=int(counts[rank]);counts[rank]+=1;assert slot<self.budget
   self.pending[key]=(rank,slot,layer+1);result.append((rank,key,slot))
  assert counts.max()-counts.min()<=1
  self.counters['issued']+=len(result)
  return result
