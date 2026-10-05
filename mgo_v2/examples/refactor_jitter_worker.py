"""Two same-policy diagnostic generations; never primary performance samples."""
from refactor_measure_worker import *
from mgo_v2.selected_runtime import create_explicit_runtime
from refactor_host_diagnostics import HostDiagnostics
import resource

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
 physical=int(os.environ['MGO_V2_PHYSICAL_GPUS'].split(',')[rank])
 a.output.mkdir(parents=True,exist_ok=True)
 from profile_watchdog import start_watchdog
 stop_watchdog=start_watchdog(a.output/f'stacks_rank{rank}.jsonl')
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42)
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 # Model loader uses cuda:0, so launch through the standard one-visible-GPU bootstrap.
 assert torch.cuda.device_count()==1
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=[3686//world+(r<3686%world) for r in range(world)];a.comm_mode='current';a.phase='COUNTERS';a.debug_plan=False
 case=json.loads(a.case.read_text());a.policy=case['policy'];horizon=case['horizon'];assert horizon==64 and world==4
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank]
 model,backing,experts=load_model();length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda');teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*len(records):(rank+1)*len(records)],device='cuda')
 rt=create_explicit_runtime(a,model,backing,experts,case);rt.stage_frozen_inputs(horizon)
 proof=json.loads((a.inputs/f'{a.policy}_P{a.arena_budget}_proof.json').read_text())
 write(a.output/f'progress_rank{rank}.json',dict(stage='WARMUP'))
 warm,expected=generate(model,rt,ids,mask,teacher,horizon);validate(rt,warm,proof,rank)
 import mgo_v2.decode_runtime as runtime_module
 import mgo_v2.pinned_h2d as h2d_module
 from contextlib import contextmanager
 original_phase=runtime_module.nvtx_phase;original_stage=h2d_module.copy_expert_to_stage
 from torch._dynamo.utils import counters
 for repeat in (1,2):
  rt.reset();a.phase='MEASURE';torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
  diag=HostDiagnostics(lambda:rt.index)
  @contextmanager
  def phase(name):
   with diag.phase(name),original_phase(name):yield
  def staging(*args,**kwargs):
   with diag.phase('moe.host_staging'):return original_stage(*args,**kwargs)
  runtime_module.nvtx_phase=phase;h2d_module.copy_expert_to_stage=staging
  uninstall=lambda:None
  if case.get('expert_diagnostics',False):
   from refactor_expert_diagnostics import install
   uninstall=install(rt,diag)
  write(a.output/f'progress_rank{rank}.json',dict(stage='DIAGNOSTIC',repeat=repeat))
  before=dict(counters['stats']);u0=resource.getrusage(resource.RUSAGE_SELF);diag.start()
  try:
   with diag.phase('generation'):
    with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
    rt.h2d.synchronize()
  finally:
   detail=diag.finish();uninstall();runtime_module.nvtx_phase=original_phase;h2d_module.copy_expert_to_stage=original_stage
  u1=resource.getrusage(resource.RUSAGE_SELF)
  assert before==dict(counters['stats']) and np.array_equal(expected,tokens)
  validate(rt,row,proof,rank)
  write(a.output/f'diagnostic_r{repeat}_rank{rank}.json',dict(status='PASS',primary_timing=False,rank=rank,repeat=repeat,case=case,metrics=row,host=detail,scheduler_metrics=dict(rt.h2d.metrics),controller_counters=dict(rt.controller.counters),process_usage={k:getattr(u1,k)-getattr(u0,k) for k in ['ru_utime','ru_stime','ru_nvcsw','ru_nivcsw']},interpretation='Two instrumented same-policy generations for phase variation, not additional primary BR-vs-LA samples. No samples excluded.'))
  dist.barrier()
 rt.close();dist.barrier();dist.destroy_process_group()
 write(a.output/f'progress_rank{rank}.json',dict(stage='COMPLETE'));stop_watchdog.set()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
