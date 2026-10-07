"""Compact exact routing-demand/Gate trace; no future routes or policy replay."""
import hashlib,json
import numpy as np
import torch.distributed as dist
class RoutingCapture:
 def __init__(self,rt,events):
  self.rt=rt;self.counts=np.zeros((events,128),np.int32);self.gates=np.zeros((events,128),np.float32);self.metrics=np.zeros((events,4),np.int64);self.seen=set();self.n=0
 def install(self):
  self.original=self.rt.controller.plan_current
  def call(event,selected,weights,origins,gates):
   assert event==self.n
   c=np.bincount(selected.ravel(),minlength=128);self.counts[event]=c;self.gates[event]=gates
   keys=event%48*128+np.flatnonzero(c);owner=self.rt.policy.owner
   hits=sum(owner[k]!=0 for k in keys);misses=[int(k) for k in keys if owner[k]==0]
   out,promotions,discards=self.original(event,selected,weights,origins,gates)
   assert not promotions and not discards
   fetches=out[5];assert len(fetches)==len(misses)
   self.metrics[event]=[hits,len(misses),sum(f[3]>=0 for f in fetches),sum(k in self.seen for k in misses)]
   self.seen.update(misses);self.n+=1
   return out,promotions,discards
  self.rt.controller.plan_current=call
 def finish(self,path):
  assert self.n==len(self.counts)
  digest=hashlib.sha256(self.counts.tobytes()+self.gates.tobytes()+self.metrics.tobytes()).hexdigest();all_hashes=[None]*4;dist.all_gather_object(all_hashes,digest);assert len(set(all_hashes))==1
  if dist.get_rank()==0:
   np.savez(path/'routing_trace.npz',counts=self.counts,gates=self.gates,reference_metrics=self.metrics,final_slots=self.rt.policy.slots)
   (path/'trace_receipt.json').write_text(json.dumps(dict(status='PASS',events=self.n,prefill_forwards=1,decode_forwards=256,local_batch=self.rt.args.local_batch,capacities=list(map(int,self.rt.args.capacities)),policy='BR',eviction='gate-score',prefetch=False,seed=42,content_sha256=digest,all_rank_agreement=True),indent=2)+'\n')
  self.rt.controller.plan_current=self.original
