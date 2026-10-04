"""Counterbalanced policy pairs with one physical C+P arena and bounded repeats."""
from refactor_measure_worker import *
from adaptive_timing import paired_decision

def main(a):
 rank=int(os.environ['RANK']);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(rank)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=[3686//8+(r<3686%8) for r in range(8)];a.comm_mode='current';a.phase='COUNTERS'
 cases=json.loads(a.cases.read_text());candidates=[c for c in cases if not c.get('baseline')];assert len(candidates)==2
 for key in ['P','trigger','horizon','overlap','partial_precision']:assert candidates[0].get(key)==candidates[1].get(key)
 baseline,candidate=[c['policy'] for c in candidates];assert baseline!=candidate;horizon=candidates[0]['horizon'];assert all(c['horizon']==horizon for c in candidates)
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records);model,backing,experts=load_model();length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda');teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda')
 first=candidates[0];a.policy=first['policy'];a.arena_budget=first['P'];a.trigger=first['trigger'];a.streaming=first['overlap'];a.ready_first=a.streaming;a.partial_precision=first.get('partial_precision');a.physical_prefetch=True;a.fused=True;a.debug_plan=False
 rt=DecodeOffloadRuntime(a,model,backing,experts);rt.stage_frozen_inputs(horizon)
 proofs={c['policy']:json.loads((a.inputs/f'{c["policy"]}_P{a.arena_budget}_proof.json').read_text()) for c in candidates};warm_tokens={};numeric={};by_policy={c['policy']:c for c in candidates}
 assert all(p['horizon']==horizon for p in proofs.values())
 def configure(policy,phase):
  a.policy=policy;a.phase=phase;rt.policy_kind={'BR':0,'CA':1,'LA':4}[policy]
  rt.proof=source['proofs'][policy];rt.reference=json.loads((a.inputs/f'{policy}_fetches.json').read_text());rt.reset()
 for policy in [baseline,candidate]:
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
 for repeat in range(1,8):
  if len(pairs)>=2 and repeat>paired_decision(pairs,baseline,candidate)['target_repeats']:break
  pair={};order=[baseline,candidate] if repeat%2 else [candidate,baseline]
  for policy in order:
   case=by_policy[policy];configure(policy,'MEASURE');torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
   from torch._dynamo.utils import counters
   before=dict(counters['stats']);key=f'{case["label"]}_r{repeat}';write(a.output/f'{key}_ready_rank{rank}.json',dict(rank=rank,case=case['label'],repeat=repeat,policy=policy));dist.barrier()
   if rank==0:write(a.output/'boundary.json',dict(key=key,case=case['label'],policy=policy,repeat=repeat,paired_order=order))
   deadline=time.monotonic()+600
   while not (a.output/f'{key}_GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('paired boundary release')
    time.sleep(.2)
   usage_before=resource.getrusage(resource.RUSAGE_SELF)
   with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
   usage_after=resource.getrusage(resource.RUSAGE_SELF);process_delta={k:getattr(usage_after,k)-getattr(usage_before,k) for k in ['ru_utime','ru_stime','ru_nvcsw','ru_nivcsw']}
   assert before==dict(counters['stats']) and np.array_equal(tokens,warm_tokens[policy]);validate(rt,row,proofs[policy],rank)
   row.update(status='PASS',rank=rank,case=case,repeat=repeat,paired_order=order,numerical_comparison=numeric[policy],no_compile_in_measure=True,process_usage_delta=process_delta,affinity=cpus,scheduler_metrics=dict(rt.h2d.metrics),controller_counters=dict(rt.controller.counters),physical_arena_count=1,peak_gpu_bytes=torch.cuda.max_memory_allocated())
   write(a.output/f'{key}_measure_rank{rank}.json',row);dist.barrier()
   pair[policy]={k:max(json.loads((a.output/f'{key}_measure_rank{r}.json').read_text())[k] for r in range(8)) for k in ['E2E_wall','TPOT']}
  pairs.append(pair)
  if rank==0:write(a.output/'phase.json',dict(stage='PAIR_COMPLETE',repeat=repeat,order=order,pairs=pairs))
 decision=paired_decision(pairs,baseline,candidate);assert decision['complete'];rt.close()
 if rank==0:write(a.output/'result.json',dict(status='PASS',baseline=baseline,candidate=candidate,cases=candidates,pairs=pairs,gate=decision,unstable=decision['unstable'],numerical_comparison=numeric,physical_arena_count=1))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
