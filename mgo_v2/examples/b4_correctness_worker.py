"""Counterbalanced policy pairs with one physical C+P arena and bounded repeats."""
from refactor_measure_worker import *
from physical_repeat_rule import decide, final_unstable
from mgo_v2.graph_expert import GraphExpertExecutor
import statistics, pickle
from mgo_v2.selected_runtime import create_explicit_runtime

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);physical=int(os.environ.get('MGO_V2_PHYSICAL_GPUS','0,1,2,3,4,5,6,7').split(',')[rank]);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=source['capacities'];assert len(a.capacities)==world;a.comm_mode='current';a.phase='COUNTERS'
 cases=json.loads(a.cases.read_text());candidates=[c for c in cases if not c.get('baseline')];assert len(candidates)==2
 for key in ['P','trigger','horizon','overlap','partial_precision','runtime_arm','staging_backend','unique_combine','async_metadata_inputs','isolated_cpu_threads','fixed_staging_team']:assert all(c.get(key)==candidates[0].get(key) for c in candidates)
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
 from mgo_v2.grouped_expert import GroupedExpertExecutor
 executor=GroupedExpertExecutor(rt.cache,rt.kernel,max_rows=batch*world*8)
 references={};results={}
 def counters_for(row):
  return dict(controller=dict(rt.controller.counters),copies=rt.h2d.metrics['copies'],bytes=rt.h2d.metrics['bytes'],forward=rt.transport.forward_bytes,back=rt.transport.return_bytes,token_hash=row['argmax_hash'],cache_keys=hashlib.sha256(rt.keys.tobytes()).hexdigest(),main_slots=hashlib.sha256(pickle.dumps(rt.policy.slots,protocol=4)).hexdigest())
 for policy in policies:
  configure(policy,'COUNTERS');rt.grouped_executor=None
  if rank==0:write(a.output/'phase.json',dict(stage='H0_FULL64_REFERENCE',policy=policy))
  reference,expected=generate(model,rt,ids,mask,teacher,horizon);validate(rt,reference,proofs[policy],rank)
  baseline=counters_for(reference)
  configure(policy,'COUNTERS');rt.grouped_executor=executor;executor.check=True
  executor.diagnostic=True;executor.records=[]
  if rank==0:write(a.output/'phase.json',dict(stage='H2_FULL64_CORRECTNESS',policy=policy))
  observed,actual=generate(model,rt,ids,mask,teacher,horizon);validate(rt,observed,proofs[policy],rank)
  actual_counters=counters_for(observed)
  parity=np.array_equal(expected,actual)
  result=dict(status='PASS' if parity and actual_counters==baseline else 'FAIL',policy=policy,rank=rank,horizon=horizon,reference=baseline,observed=actual_counters,token_parity=bool(parity),token_mismatches=int(np.count_nonzero(expected!=actual)),executor=executor.receipt(),peak_gpu_bytes=torch.cuda.max_memory_allocated())
  write(a.output/f'{policy}_correctness_rank{rank}.json',result)
  write(a.output/f'{policy}_waves_rank{rank}.json',executor.records)
  results[policy]=result
  dist.barrier()
  # Every rank writes evidence before making the collective gate decision.
  all_rows=[json.loads((a.output/f'{policy}_correctness_rank{r}.json').read_text()) for r in range(world)]
  if any(r['status']!='PASS' for r in all_rows):break
 rt.close();dist.barrier()
 if rank==0:
  rows=[json.loads(p.read_text()) for p in a.output.glob('*_correctness_rank*.json')]
  write(a.output/'result.json',dict(status='PASS',correctness_pass=len(rows)==8 and all(r['status']=='PASS' for r in rows),purpose='B4 correctness only; no primary timing',ranks=rows))
 dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
