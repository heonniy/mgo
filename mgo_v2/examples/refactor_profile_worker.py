"""Separate Nsight capture; never used as primary physical timing evidence."""
from refactor_measure_worker import *

class ProfileRuntime(DecodeOffloadRuntime):
 def execute(self,*args):
  if not getattr(self,'capture',False):return super().execute(*args)
  with nvtx_phase(f'{"decode" if self.index>=48 else "prefill"}.event.{self.index}'):
   result=super().execute(*args)
  if self.index==48:self.profile_prefill_metrics=dict(self.h2d.metrics)
  return result

 def arm_profile(self):
  scheduler=self.h2d;scheduler.profile=True;self.profile_tickets=[]
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
   row.update(kind='demand' if row['origin']=='demand' else 'useful_prefetch' if row['use_event'] is not None else 'wasted_prefetch',bytes=9437184,event_dma_ms=t.begin.elapsed_time(t.done))
   rows.append(row)
  assert len(rows)==self.h2d.metrics['copies']-self.profile_prefill_metrics['copies']
  return rows

def main(a):
 rank=int(os.environ['RANK']);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(rank)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=[3686//8+(r<3686%8) for r in range(8)];a.comm_mode='current'
 case=json.loads(a.case.read_text());horizon=case['horizon'];assert horizon==8 and case['partial_precision']=='bf16'
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
 before=dict(counters['stats']);torch.cuda.profiler.start()
 try:
  with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
  torch.cuda.synchronize();rt.h2d.synchronize()
 finally:
  torch.cuda.profiler.stop();h2d_module.copy_expert_to_stage=original_stage
 assert before==dict(counters['stats']) and np.array_equal(expected,tokens);validate(rt,row,proof,rank)
 trace_path=a.output/f'copy_trace_rank{rank}.json';write(trace_path,rt.copy_trace())
 write(a.output/f'rank{rank}.json',dict(status='PASS',purpose='instrumented diagnostic; not primary timing',copy_trace_path=str(trace_path),copy_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),case=case,no_compile_in_capture=True,scheduler_metrics=rt.h2d.metrics,decode_expert_copies=rt.h2d.metrics['copies']-rt.profile_prefill_metrics['copies'],decode_expert_bytes=rt.h2d.metrics['bytes']-rt.profile_prefill_metrics['bytes'],controller_counters=rt.controller.counters,transport_calls=rt.transport.calls,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
 rt.close();dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
