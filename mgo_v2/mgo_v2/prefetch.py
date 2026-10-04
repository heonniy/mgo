"""Rank-local transfer state; logical plans live in controller.py."""
# Compatibility name for CPU characterization; there is one implementation.
from .controller import DecodePrefetchController as PrefetchShadow

class TransferState:
 """Rank-local transfer state, deliberately excluded from replicated shadows."""
 def __init__(self,key):
  self.key=key;self.state='QUEUED';self.urgent=False;self.discard_after_copy=False
 def demand(self):
  if self.state=='EMPTY':raise RuntimeError('demanding canceled transfer')
  self.urgent=True
 def start(self):
  if self.state!='QUEUED':raise RuntimeError('transfer already started/canceled')
  self.state='INFLIGHT'
 def complete(self):
  if self.state!='INFLIGHT':raise RuntimeError('completion without active DMA')
  self.state='EMPTY' if self.discard_after_copy else 'READY'
 def discard(self):
  if self.urgent:raise RuntimeError('cannot discard demanded transfer')
  if self.state=='INFLIGHT':self.discard_after_copy=True
  else:self.state='EMPTY'
