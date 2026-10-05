"""Separate Nsight capture; never used as primary physical timing evidence."""
from refactor_measure_worker import *

class ProfileRuntime(DecodeOffloadRuntime):
 def plan_event(self,*args):
  event=super().plan_event(*args)
  if getattr(self,'capture',False) and self.index>=48:
   sent=list(map(int,event['send_counts']));received=list(map(int,event['recv_counts']))
   self.profile_communication.append(dict(event=self.index,send_token_rows=sent,recv_token_rows=received,
    active_remote_send_peers=sum(n>0 for r,n in enumerate(sent) if r!=self.rank),
    active_remote_recv_peers=sum(n>0 for r,n in enumerate(received) if r!=self.rank),
    forward_wire_bytes=(sum(sent)-sent[self.rank])*(2048*2+24),
    return_wire_bytes=(sum(received)-received[self.rank])*2048*2))
  return event

 def execute(self,*args):
  if not getattr(self,'capture',False):return super().execute(*args)
  with nvtx_phase(f'{"decode" if self.index>=48 else "prefill"}.event.{self.index}'):
   result=super().execute(*args)
  if self.index%192==0:
   write(self.args.output/f'progress_rank{self.rank}.json',dict(stage='CAPTURE',completed_events=self.index,unix=time.time()))
  if self.index==48:self.profile_prefill_metrics=dict(self.h2d.metrics)
  return result

 def arm_profile(self):
  scheduler=self.h2d;scheduler.profile=True;self.profile_tickets=[];self.profile_communication=[]
  original_predict=self.predictor.predict_next
  def predict(*args,**kwargs):
   with nvtx_phase('moe.prefetch_predictor'):return original_predict(*args,**kwargs)
  self.predictor.predict_next=predict
  for method,origin in [('enqueue_demand','demand'),('enqueue_prefetch','prefetch')]:
   original=getattr(scheduler,method)
   def enqueue(slot,key,tensors,_original=original,_origin=origin):
    event=self.index;t=_original(slot,key,tensors)
    if not hasattr(t,'profile_meta'):
     t.profile_meta=dict(source_event=event,key=int(key),origin=_origin,use_event=None,discard_event=None,readiness_at_use=None);self.profile_tickets.append(t)
    return t
   setattr(scheduler,method,enqueue)
  original_promote=scheduler.promote
  def promote(slot,key):
   with scheduler.cv:
    t=scheduler.tickets[slot]
    readiness='ready' if t.submitted and t.done.query() else 'inflight' if t.submitted else 'queued'
    t.profile_meta.update(use_event=self.index,readiness_at_use=readiness)
   return original_promote(slot,key)
  scheduler.promote=promote
  original_discard=scheduler.discard
  def discard(slot,key):
   t=scheduler.tickets.get(slot)
   if t is not None and t.key==key:t.profile_meta['discard_event']=self.index
   return original_discard(slot,key)
  scheduler.discard=discard
 def copy_trace(self):
  rows=[]
  for t in self.h2d.trace:
   if t.profile_meta['source_event']<48:continue
   row=dict(t.profile_meta)
   row.update(kind='demand' if row['origin']=='demand' else 'useful_prefetch' if row['use_event'] is not None else 'wasted_prefetch',bytes=9437184,event_dma_ms=t.begin.elapsed_time(t.done),readiness_observation='Logical promotion in current-layer controller, before forward payload launch; not the later expert kernel start.')
   rows.append(row)
  assert len(rows)==self.h2d.metrics['copies']-self.profile_prefill_metrics['copies']
  return rows

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);physical=int(os.environ.get('MGO_V2_PHYSICAL_GPUS','0,1,2,3,4,5,6,7').split(',')[rank]);a.output.mkdir(parents=True,exist_ok=True)
 import faulthandler
 stack_log=(a.output/f'stacks_rank{rank}.log').open('w')
 faulthandler.enable(file=stack_log)
 faulthandler.dump_traceback_later(120,repeat=True,file=stack_log)
 def progress(stage):write(a.output/f'progress_rank{rank}.json',dict(stage=stage,unix=time.time()))
 progress('SETUP')
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=[3686//world+(r<3686%world) for r in range(world)];a.comm_mode='current'
 case=json.loads(a.case.read_text());horizon=case['horizon'];assert horizon in (8,64,256) and case['partial_precision']=='bf16'
 a.staging_backend=case.get('staging_backend','torch');assert a.staging_backend in ('torch','memmove')
 a.unique_combine=case.get('unique_combine',False);assert type(a.unique_combine) is bool
 a.policy=case['policy'];a.arena_budget=case['P'];a.trigger=case['trigger'];a.partial_precision='bf16';a.streaming=case['overlap'];a.ready_first=a.streaming;a.physical_prefetch=True;a.fused=True;a.debug_plan=False;a.phase='COUNTERS'
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records);model,backing,experts=load_model();length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda');teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda')
 proof=json.loads((a.inputs/f'{a.policy}_P{a.arena_budget}_proof.json').read_text());assert proof['horizon']==horizon
 rt=ProfileRuntime(a,model,backing,experts);rt.stage_frozen_inputs(horizon)
 warm,expected=generate(model,rt,ids,mask,teacher,horizon);validate(rt,warm,proof,rank)
 rt.reset();a.phase='MEASURE';rt.capture=True;rt.arm_profile();torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
 from torch._dynamo.utils import counters
 # Attribute pageable-to-pinned staging on its actual worker thread. This
 # monkeypatch exists only in this dedicated diagnostic process, never timing.
 import mgo_v2.pinned_h2d as h2d_module
 original_stage=h2d_module.copy_expert_to_stage
 def profiled_stage(*args,**kwargs):
  with nvtx_phase('moe.host_staging'):return original_stage(*args,**kwargs)
 h2d_module.copy_expert_to_stage=profiled_stage
 import mgo_v2.decode_runtime as runtime_module
 diagnostic_hooks=[]
 for module,name,phase in [(dist,'all_gather_into_tensor','moe.metadata_collective_submit'),
                           (runtime_module,'plan_layout','moe.packet_layout_cpu'),
                           (runtime_module,'device_layout','moe.packet_layout_upload')]:
  original=getattr(module,name);diagnostic_hooks.append((module,name,original))
  def profiled_call(*args,_original=original,_phase=phase,**kwargs):
   with nvtx_phase(_phase):return _original(*args,**kwargs)
  setattr(module,name,profiled_call)
 before=dict(counters['stats']);progress('PROFILER_START');torch.cuda.profiler.start();progress('CAPTURE_STARTED')
 try:
  with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
  progress('CAPTURE_GENERATION_RETURNED');torch.cuda.synchronize();rt.h2d.synchronize()
 finally:
  progress('PROFILER_STOP');torch.cuda.profiler.stop();progress('PROFILER_STOPPED');h2d_module.copy_expert_to_stage=original_stage
  for module,name,original in diagnostic_hooks:setattr(module,name,original)
 assert before==dict(counters['stats']) and np.array_equal(expected,tokens);validate(rt,row,proof,rank)
 trace_path=a.output/f'copy_trace_rank{rank}.json';write(trace_path,rt.copy_trace())
 assert len(rt.profile_communication)==48*horizon and rt.transport.calls==96*horizon
 assert sum(r['forward_wire_bytes'] for r in rt.profile_communication)==rt.transport.forward_bytes
 assert sum(r['return_wire_bytes'] for r in rt.profile_communication)==rt.transport.return_bytes
 write(a.output/f'communication_rank{rank}.json',dict(status='PASS',rank=rank,events=rt.profile_communication,metadata_calls=rt.metadata.calls,payload_calls=rt.transport.calls,forward_wire_bytes=rt.transport.forward_bytes,return_wire_bytes=rt.transport.return_bytes,scope='Decode BF16 token/rank payload bytes excluding self, protocol overhead and metadata. Per-event peer counts come from the executed layout.'))
 write(a.output/f'rank{rank}.json',dict(status='PASS',rank=rank,purpose='instrumented diagnostic; not primary timing',copy_trace_path=str(trace_path),copy_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),case=case,no_compile_in_capture=True,scheduler_metrics=rt.h2d.metrics,decode_expert_copies=rt.h2d.metrics['copies']-rt.profile_prefill_metrics['copies'],decode_expert_bytes=rt.h2d.metrics['bytes']-rt.profile_prefill_metrics['bytes'],controller_counters=rt.controller.counters,transport_calls=rt.transport.calls,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
 rt.close();dist.barrier();dist.destroy_process_group()
 progress('COMPLETE');faulthandler.cancel_dump_traceback_later();stack_log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
