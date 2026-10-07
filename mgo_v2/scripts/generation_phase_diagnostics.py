"""Separate full-generation diagnostic using existing non-synchronizing events.

Current-stream spans include submission gaps and peer waits. Copy-stream
service overlaps them and must not be added to the partition.
"""
from prefill_phase_diagnostics import PrefillDiagnostics

class GenerationDiagnostics(PrefillDiagnostics):
 def mark(self):
  super().mark()
  self.rows[-1].update(event_index=self.rt.index,step=self.rt.index//48,layer=self.rt.index%48)
 def install(self,model):
  super().install(model)
  self.wrap(self.rt.h2d,'wait_for_slot','required_h2d_exposed_wait')
  self.cache_events=[];self.prefill_survivors=set()
  original=self.rt.plan_event
  def plan(layer,*args,**kwargs):
   rt=self.rt;idx=rt.index
   before=set(int(k) for k in rt.policy.slots[rt.rank] if k>=0)
   if idx==48:self.prefill_survivors=before.copy()
   event=original(layer,*args,**kwargs)
   fetched={int(k) for k,_,_,_ in event['fetches']}
   after=set(int(k) for k in rt.policy.slots[rt.rank] if k>=0)
   self.prefill_survivors.intersection_update(after);self.prefill_survivors.difference_update(fetched)
   if idx>=48:
    counts={name:0 for name in ('expert_uses','token_expert_uses','main_hits','prefetch_hits','demand_misses','main_hit_tokens','prefetch_hit_tokens','miss_tokens','surviving_prefill_hits','surviving_prefill_hit_tokens')}
    for expert,rows,cols,slot in event['groups']:
     key=layer*128+expert;n=len(rows);counts['expert_uses']+=1;counts['token_expert_uses']+=n
     if key in fetched:counts['demand_misses']+=1;counts['miss_tokens']+=n
     elif key in before:
      counts['main_hits']+=1;counts['main_hit_tokens']+=n
      if key in self.prefill_survivors:counts['surviving_prefill_hits']+=1;counts['surviving_prefill_hit_tokens']+=n
     else:counts['prefetch_hits']+=1;counts['prefetch_hit_tokens']+=n
    assert counts['expert_uses']==counts['main_hits']+counts['prefetch_hits']+counts['demand_misses']
    self.cache_events.append(dict(event_index=idx,step=idx//48,layer=layer,**counts))
   return event
  self.rt.plan_event=plan;self.undo.append(lambda:setattr(self.rt,'plan_event',original))
 def finish(self,wall_seconds):
  # Drain background prefetch outside the measured diagnostic boundary.
  self.rt.h2d.synchronize()
  result=super().finish(wall_seconds)
  result['decode_cache_events']=self.cache_events
  result['cache_definition']='Logical main hits and promoted-prefetch hits avoid a new demand copy, but can still wait for an in-flight H2D. Prefill survivors exclude keys evicted or demand-reloaded on this rank.'
  return result
