"""Separate full-generation MoE boundaries and host collective-entry clocks."""
import torch
from generation_phase_diagnostics import GenerationDiagnostics
class PrefetchDiagnostics(GenerationDiagnostics):
 def mark(self):
  super().mark();self.rows[-1]['host_timestamp_ns']=self.wall
 def install(self,model):
  super().install(model);self.moe_events=[];self.open_moe={}
  def pre(module,*unused):
   if self.active:
    start=torch.cuda.Event(enable_timing=True);start.record()
    import time
    self.open_moe[module]=(self.rt.index,start,time.perf_counter_ns())
  def post(module,*unused):
   if self.active:
    import time
    index,start,cpu_start=self.open_moe.pop(module);end=torch.cuda.Event(enable_timing=True);end.record()
    self.moe_events.append((index,start,end,cpu_start,time.perf_counter_ns()))
  for block in model.model.layers:
   # prepend places the start before the router diagnostic hook.
   self.hooks.append(block.mlp.register_forward_pre_hook(pre,prepend=True))
   self.hooks.append(block.mlp.register_forward_hook(post))
 def finish(self,wall_seconds):
  result=super().finish(wall_seconds)
  assert not self.open_moe
  result['moe_events']=[dict(event_index=i,step=i//48,layer=i%48,stream_seconds=a.elapsed_time(b)/1000,host_start_ns=c,host_end_ns=d) for i,a,b,c,d in self.moe_events]
  assert len(result['moe_events'])==257*48
  result['moe_definition']='CUDA current-stream MLP boundary includes router, metadata, placement, dispatch, H2D dependencies, expert, partial, return and combine. Excludes attention and layernorm outside MLP. Includes CPU submission gaps and peer waits; not pure kernel time.'
  return result
