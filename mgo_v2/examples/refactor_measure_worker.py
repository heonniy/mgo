"""Full-model frozen-route timing, live cache strategy, external clean boundaries."""
from refactor_baseline_worker import *
from mgo_v2.decode_runtime import DecodeOffloadRuntime

def validate(rt,row,proof,rank):
 rt.h2d.synchronize()
 assert not rt.mismatch.item()
 if isinstance(rt,DecodeOffloadRuntime):
  assert array_hash(rt.policy.slots[rank,:rt.cap])==proof['rank_state_hashes'][rank]
  assert array_hash(rt.arena.main_physical[rank])==proof['rank_role_hashes'][rank]
  assert rt.controller.counters==proof['counters']
  assert rt.h2d.metrics['copies']+rt.h2d.metrics['canceled']==proof['max_copy_counts'][rank]
  assert rt.transport.calls==proof['horizon']*48*2
 else:
  assert array_hash(rt.keys)==proof['rank_state_hashes'][rank]
  assert rt.actual_h2d_bytes==proof['H2D_bytes'][rank]

def main(a):
 rank=int(os.environ['RANK']);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(rank)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=[3686//8+(r<3686%8) for r in range(8)];a.comm_mode='current'
 cases=json.loads(a.cases.read_text());records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records)
 model,backing,experts=load_model();length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
 teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda');references={};results=[]
 for case in cases:
  label=case['label'];horizon=case['horizon'];a.policy=case['policy'];a.arena_budget=case['P'];a.trigger=case['trigger'];a.streaming=case.get('overlap',True);a.ready_first=a.streaming;a.physical_prefetch=True;a.fused=True;a.debug_plan=False;a.phase='COUNTERS'
  proof=json.loads((a.inputs/f'{a.policy}_P{a.arena_budget}_proof.json').read_text());assert proof['horizon']==horizon
  baseline=case.get('baseline',False);rt=(PrefixRuntime if baseline else DecodeOffloadRuntime)(a,model,backing,experts)
  if not baseline:rt.stage_frozen_inputs(horizon)
  if rank==0:write(a.output/'phase.json',dict(stage='WARMUP_VALIDATION',case=label,case_spec=case))
  warm,tokens=generate(model,rt,ids,mask,teacher,horizon);validate(rt,warm,proof,rank)
  if baseline:
   references[horizon]=warm['argmax_hash']
   if horizon==256:
    old=Path('/home/hwlee/mgo-results/la_physical_validation_20261004')/f'B{batch}_BR_env1'/f'validation_rank{rank}.json'
    assert warm['argmax_hash']==json.loads(old.read_text())['argmax_hash']
  else:assert warm['argmax_hash']==references[horizon],('reference output mismatch',label,rank)
  write(a.output/f'{label}_validation_rank{rank}.json',dict(status='PASS',rank=rank,case=case,argmax_hash=warm['argmax_hash'],state_hash=warm['state_hash'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),frozen_input_bytes=getattr(rt,'frozen_input_bytes',0),scheduler_metrics=(rt.h2d.metrics if not baseline else None)))
  if case.get('measure',True):
   samples=[]
   for repeat in (1,2,3):
    if repeat==3 and decide(samples)['target_repeats']!=3:break
    a.phase='MEASURE';rt.reset();torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
    from torch._dynamo.utils import counters
    before=dict(counters['stats']);key=f'{label}_r{repeat}'
    write(a.output/f'{key}_ready_rank{rank}.json',dict(rank=rank,case=label,repeat=repeat));dist.barrier()
    if rank==0:write(a.output/'boundary.json',dict(key=key,case=label,repeat=repeat))
    deadline=time.monotonic()+600
    while not (a.output/f'{key}_GO').exists():
     if time.monotonic()>deadline:raise TimeoutError('boundary release')
     time.sleep(.2)
    with torch._dynamo.config.patch(error_on_recompile=True):row,actual=generate(model,rt,ids,mask,teacher,horizon)
    assert before==dict(counters['stats']) and np.array_equal(tokens,actual)
    validate(rt,row,proof,rank)
    row.update(status='PASS',rank=rank,case=case,repeat=repeat,no_compile_in_measure=True,scheduler_metrics=dict(rt.h2d.metrics),controller_counters=dict(rt.controller.counters),peak_gpu_bytes=torch.cuda.max_memory_allocated())
    write(a.output/f'{key}_measure_rank{rank}.json',row);dist.barrier()
    samples.append({k:max(json.loads((a.output/f'{key}_measure_rank{r}.json').read_text())[k] for r in range(8)) for k in ['E2E_wall','TPOT']})
    if rank==0:write(a.output/'phase.json',dict(stage='MEASURE_COMPLETE',case=label,repeat=repeat,samples=samples))
   result=dict(status='PASS',case=case,samples=samples,gate=decide(samples[:2]),unstable=final_unstable(samples));results.append(result)
   if rank==0:write(a.output/f'{label}_result.json',result)
  if hasattr(rt,'close'):rt.close()
  for block in model.model.layers:block.mlp.forward=lambda *args:None
  del rt;gc.collect();torch.cuda.empty_cache();dist.barrier()
 if rank==0:write(a.output/'result.json',dict(status='PASS',results=results))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
