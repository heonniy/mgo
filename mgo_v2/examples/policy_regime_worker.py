"""Counterbalanced policy pairs with one physical C+P arena and bounded repeats."""
from refactor_measure_worker import *
from adaptive_timing import paired_decision
from mgo_v2.selected_runtime import create_explicit_runtime

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);physical=int(os.environ.get('MGO_V2_PHYSICAL_GPUS','0,1,2,3,4,5,6,7').split(',')[rank]);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=source['capacities'];assert len(a.capacities)==world;a.comm_mode='current';a.phase='COUNTERS'
 cases=json.loads(a.cases.read_text());candidates=[c for c in cases if not c.get('baseline')];assert len(candidates)==4
 for key in ['P','trigger','horizon','overlap','partial_precision','runtime_arm','staging_backend','unique_combine','async_metadata_inputs','isolated_cpu_threads','fixed_staging_team']:assert all(c.get(key)==candidates[0].get(key) for c in candidates)
 policies=[c['policy'] for c in candidates];baseline=policies[0];assert policies==['BR','OLD_CA','FCA','LA_CA'];horizon=candidates[0]['horizon'];assert all(c['horizon']==horizon for c in candidates)
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records);model,backing,experts=load_model();length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda');teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda')
 first=candidates[0];a.policy=first['policy'];a.debug_plan=False
 assert not first.get('fixed_staging_team',False) or first.get('isolated_cpu_threads',False)
 a.staging_cpu_team=cpus[1:3] if first.get('fixed_staging_team',False) else None
 rt=create_explicit_runtime(a,model,backing,experts,first);rt.stage_frozen_inputs(horizon)
 assert rt.metadata.async_inputs==first.get('async_metadata_inputs',False)
 proofs={c['policy']:json.loads((a.inputs/f'{c["policy"]}_P{a.arena_budget}_proof.json').read_text()) for c in candidates};warm_tokens={};numeric={};by_policy={c['policy']:c for c in candidates}
 assert all(p['horizon']==horizon for p in proofs.values())
 def configure(policy,phase):
  a.policy=policy;a.phase=phase;rt.policy_kind={'BR':0,'OLD_CA':3,'FCA':5,'LA_CA':6}[policy]
  rt.proof=source['proofs'][policy];rt.reference=json.loads((a.inputs/f'{policy}_fetches.json').read_text());rt.reset()
  rt.measurement_thread_placement=None
  if first.get('isolated_cpu_threads',False):
   from refactor_thread_affinity import configure as configure_affinity
   rt.measurement_thread_placement=configure_affinity(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
 for policy in policies:
  configure(policy,'COUNTERS')
  if rank==0:write(a.output/'phase.json',dict(stage='WARMUP_VALIDATION',policy=policy,case=by_policy[policy]))
  nonfinite=torch.zeros((),device='cuda',dtype=torch.bool)
  def check_finite(module,inputs,output):nonfinite.logical_or_(~torch.isfinite(output.logits).all())
  hook=model.register_forward_hook(check_finite)
  try:row,tokens=generate(model,rt,ids,mask,teacher,horizon)
  finally:hook.remove()
  assert not nonfinite.item();validate(rt,row,proofs[policy],rank)
  assert a.partial_precision=='bf16','owner selected BF16-only paired study'
  warm_tokens[policy]=tokens.copy();numeric[policy]=dict(actual_argmax_hash=row['argmax_hash'],partial_precision=a.partial_precision,legacy_comparison='not collected: owner BF16-only amendment')
  write(a.output/f'{policy}_validation_rank{rank}.json',dict(status='PASS',rank=rank,case=by_policy[policy],numerical_comparison=numeric[policy],finite_logits=True,physical_arena_count=1,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
 pairs=[]
 for repeat in range(1,4):
  if len(pairs)>=2 and all(paired_decision(pairs,baseline,p)['target_repeats']<repeat for p in policies[1:]):break
  pair={};order=policies if repeat%2 else list(reversed(policies))
  for policy in order:
   case=by_policy[policy];configure(policy,'MEASURE');torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
   from torch._dynamo.utils import counters
   before=dict(counters['stats']);key=f'{case["label"]}_r{repeat}';write(a.output/f'{key}_ready_rank{rank}.json',dict(rank=rank,case=case['label'],repeat=repeat,policy=policy));dist.barrier()
   if rank==0:write(a.output/'boundary.json',dict(key=key,case=case['label'],policy=policy,repeat=repeat,paired_order=order))
   deadline=time.monotonic()+600
   while not (a.output/f'{key}_GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('paired boundary release')
    time.sleep(.2)
   thread_before=refactor_thread_usage.snapshot(rt.h2d.thread.native_id)
   usage_before=resource.getrusage(resource.RUSAGE_SELF)
   with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
   usage_after=resource.getrusage(resource.RUSAGE_SELF);process_delta={k:getattr(usage_after,k)-getattr(usage_before,k) for k in ['ru_utime','ru_stime','ru_nvcsw','ru_nivcsw']}
   thread_delta=refactor_thread_usage.delta(thread_before,refactor_thread_usage.snapshot(rt.h2d.thread.native_id))
   assert before==dict(counters['stats']) and np.array_equal(tokens,warm_tokens[policy]);validate(rt,row,proofs[policy],rank)
   row.update(status='PASS',rank=rank,case=case,async_metadata_inputs=rt.metadata.async_inputs,repeat=repeat,paired_order=order,numerical_comparison=numeric[policy],no_compile_in_measure=True,thread_placement=rt.measurement_thread_placement,process_usage_delta=process_delta,thread_usage_delta=thread_delta,affinity=cpus,scheduler_metrics=dict(rt.h2d.metrics),controller_counters=dict(rt.controller.counters),physical_arena_count=1,peak_gpu_bytes=torch.cuda.max_memory_allocated())
   write(a.output/f'{key}_measure_rank{rank}.json',row);dist.barrier()
   pair[policy]={k:max(json.loads((a.output/f'{key}_measure_rank{r}.json').read_text())[k] for r in range(world)) for k in ['E2E_wall','TPOT']}
  pairs.append(pair)
  if rank==0:write(a.output/'phase.json',dict(stage='PAIR_COMPLETE',repeat=repeat,order=order,pairs=pairs))
 decisions={p:paired_decision(pairs,baseline,p) for p in policies[1:]};assert all(d['complete'] for d in decisions.values());rt.close()
 if rank==0:write(a.output/'result.json',dict(status='PASS',baseline=baseline,candidates=policies[1:],cases=candidates,pairs=pairs,gates=decisions,unstable=any(d['unstable'] for d in decisions.values()),numerical_comparison=numeric,physical_arena_count=1))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
