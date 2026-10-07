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
 def finish(self,wall_seconds):
  # Drain background prefetch outside the measured diagnostic boundary.
  self.rt.h2d.synchronize()
  return super().finish(wall_seconds)
