"""Separate Nsight capture; never used as primary physical timing evidence."""
from refactor_measure_worker import *

class ProfileRuntime(DecodeOffloadRuntime):
 def execute(self,*args):
  if not getattr(self,'capture',False):return super().execute(*args)
  with nvtx_phase(f'{"decode" if self.index>=48 else "prefill"}.event.{self.index}'):
   return super().execute(*args)

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
 rt.reset();a.phase='MEASURE';rt.capture=True;torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
 from torch._dynamo.utils import counters
 before=dict(counters['stats']);torch.cuda.profiler.start()
 try:
  with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
  torch.cuda.synchronize();rt.h2d.synchronize()
 finally:torch.cuda.profiler.stop()
 assert before==dict(counters['stats']) and np.array_equal(expected,tokens);validate(rt,row,proof,rank)
 write(a.output/f'rank{rank}.json',dict(status='PASS',purpose='instrumented diagnostic; not primary timing',case=case,no_compile_in_capture=True,scheduler_metrics=rt.h2d.metrics,controller_counters=rt.controller.counters,transport_calls=rt.transport.calls,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
 rt.close();dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
