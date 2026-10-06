"""Counterbalanced policy pairs with one physical C+P arena and bounded repeats."""
from refactor_measure_worker import *
from physical_repeat_rule import decide, final_unstable
from mgo_v2.graph_expert import GraphExpertExecutor
import statistics
from mgo_v2.selected_runtime import create_explicit_runtime

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);physical=int(os.environ.get('MGO_V2_PHYSICAL_GPUS','0,1,2,3,4,5,6,7').split(',')[rank]);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=source['capacities'];assert len(a.capacities)==world;a.comm_mode='current';a.phase='COUNTERS'
 cases=json.loads(a.cases.read_text());candidates=[c for c in cases if not c.get('baseline')];assert len(candidates)==2
 for key in ['P','trigger','horizon','overlap','partial_precision','runtime_arm','staging_backend','unique_combine','async_metadata_inputs','isolated_cpu_threads','fixed_staging_team','post_expert_barrier']:assert all(c.get(key)==candidates[0].get(key) for c in candidates)
 policies=[c['policy'] for c in candidates];baseline=policies[0];assert policies==['BR','FCA'];horizon=candidates[0]['horizon'];assert all(c['horizon']==horizon for c in candidates)
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
 candidate=first.get('executor','H1');assert candidate in ('H1','H1b')
 graph=GraphExpertExecutor(rt.cache,rt.kernel,wrapper=candidate=='H1b')
 rt.graph_executor=graph
 references={}
 for policy in policies:
  configure(policy,'COUNTERS')
  if rank==0:write(a.output/'phase.json',dict(stage='FULL64_H0_DISCOVERY',policy=policy))
  row,tokens=generate(model,rt,ids,mask,teacher,horizon);validate(rt,row,proofs[policy],rank)
  warm_tokens[policy]=tokens.copy()
  expected_barriers=horizon*48 if first.get('post_expert_barrier',False) else 0
  assert rt.post_expert_barriers==expected_barriers,(policy,rt.post_expert_barriers,expected_barriers)
  references[policy]=dict(counters=dict(rt.controller.counters),copies=rt.h2d.metrics['copies'],bytes=rt.h2d.metrics['bytes'],forward=rt.transport.forward_bytes,back=rt.transport.return_bytes,post_expert_barriers=rt.post_expert_barriers)
 rt.h2d.synchronize();torch.cuda.synchronize();rt.h2d.close()
 def graph_progress(row):write(a.output/f'graph_progress_rank{rank}.json',row)
 graph.build(graph_progress)
 write(a.output/f'graph_signatures_rank{rank}.json',graph.receipt())
 for policy in policies:
  configure(policy,'COUNTERS');rt.graph_check=True
  if rank==0:write(a.output/'phase.json',dict(stage='FULL64_H1_VALIDATION',policy=policy))
  from torch._dynamo.utils import counters
  before=dict(counters['stats'])
  with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
  validate(rt,row,proofs[policy],rank)
  assert before==dict(counters['stats']) and np.array_equal(tokens,warm_tokens[policy])
  observed=dict(counters=dict(rt.controller.counters),copies=rt.h2d.metrics['copies'],bytes=rt.h2d.metrics['bytes'],forward=rt.transport.forward_bytes,back=rt.transport.return_bytes,post_expert_barriers=rt.post_expert_barriers)
  assert observed==references[policy]
  write(a.output/f'{policy}_correctness_rank{rank}.json',dict(status='PASS',reference=references[policy],observed=observed,argmax_hash=row['argmax_hash'],graph=graph.receipt()))
 rt.graph_check=False
 samples={key:[] for key in ['BR_H0','FCA_H0','BR_'+candidate,'FCA_'+candidate]}
 for repeat in (1,2,3):
  order=list(samples) if repeat%2 else list(reversed(samples))
  for key in order:
   if repeat==3 and decide(samples[key])['target_repeats']!=3:continue
   policy,mode=key.split('_');rt.graph_executor=graph if mode==candidate else None
   configure(policy,'MEASURE');torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
   before=dict(counters['stats']);label=f'{key}_r{repeat}'
   write(a.output/f'{label}_ready_rank{rank}.json',dict(rank=rank,case=key,repeat=repeat));dist.barrier()
   if rank==0:write(a.output/'boundary.json',dict(key=label,case=key,repeat=repeat,paired_order=order))
   deadline=time.monotonic()+600
   while not (a.output/f'{label}_GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('B3 clean boundary release')
    time.sleep(.2)
   with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
   assert before==dict(counters['stats']) and np.array_equal(tokens,warm_tokens[policy])
   validate(rt,row,proofs[policy],rank)
   observed=dict(counters=dict(rt.controller.counters),copies=rt.h2d.metrics['copies'],bytes=rt.h2d.metrics['bytes'],forward=rt.transport.forward_bytes,back=rt.transport.return_bytes,post_expert_barriers=rt.post_expert_barriers)
   assert observed==references[policy]
   row.update(status='PASS',rank=rank,policy=policy,executor=mode,repeat=repeat,no_compile_in_measure=True,thread_placement=rt.measurement_thread_placement,counters=observed,peak_gpu_bytes=torch.cuda.max_memory_allocated())
   write(a.output/f'{label}_measure_rank{rank}.json',row);dist.barrier()
   samples[key].append({k:max(json.loads((a.output/f'{label}_measure_rank{r}.json').read_text())[k] for r in range(world)) for k in ['E2E_wall','TPOT']})
   if rank==0:write(a.output/'phase.json',dict(stage='MEASURE_COMPLETE',case=key,repeat=repeat,samples=samples))
 gates={key:dict(initial=decide(rows[:2]),unstable=final_unstable(rows),estimator='mean_and_median' if len(rows)==2 else 'median',summary={k:dict(n=len(rows),mean=statistics.mean(r[k] for r in rows),median=statistics.median(r[k] for r in rows),minimum=min(r[k] for r in rows),maximum=max(r[k] for r in rows)) for k in ['E2E_wall','TPOT']},estimate={k:statistics.median([r[k] for r in rows]) for k in ['E2E_wall','TPOT']},range={k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in ['E2E_wall','TPOT']}) for key,rows in samples.items()}
 rt.close()
 if rank==0:write(a.output/'result.json',dict(status='PASS',candidate=candidate,samples=samples,gates=gates,unstable=any(g['unstable'] for g in gates.values()),rule='Two repeats; <=2% stop; >2% and <=5% one third; >5% unstable without extra repeat. Retain every sample.'))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
