"""Ready-first traversal preserves the original result positions."""
def expert_order(groups,scheduler,ready_first=False,metrics=None):
 pending=list(range(len(groups)));before_wait=0;waited=False
 while pending:
  chosen=pending[0]
  if ready_first:
   chosen=next((i for i in pending if scheduler.ready(groups[i][-1])),None)
  if chosen is None:
   chosen=pending[0]
   if metrics is not None:
    metrics['waits']+=1
    if not waited:metrics['ready_before_first_wait']+=before_wait
   waited=True;scheduler.wait_for_slot(groups[chosen][-1])
  elif not ready_first:scheduler.wait_for_slot(groups[chosen][-1])
  else:before_wait+=not waited
  pending.remove(chosen);yield chosen
 if metrics is not None and not waited:metrics['ready_before_first_wait']+=before_wait
