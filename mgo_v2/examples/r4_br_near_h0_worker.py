"""Four frozen R4 workloads, BR/Near, selected H0/full-pinned runtime."""
from refactor_measure_worker import *
from mgo_v2.selected_runtime import create_selected_runtime,selected_options
from mgo_v2.decode_runtime import DecodeOffloadRuntime
import resource
POLICIES=('BR','LA_CA_NEAR')
METRICS=('TTFT','TPOT','E2E')
def generation(model,rt,initial,teacher):
 ids=initial;mask=torch.ones_like(ids);past=None;tokens=[];times=[];finite=torch.ones((),dtype=torch.bool,device='cuda')
 dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():
  for step in range(33):
   position=torch.arange(mask.shape[1]-ids.shape[1],mask.shape[1],device='cuda')[None,:].expand(len(ids),-1)
   out=model(input_ids=ids,attention_mask=mask,position_ids=position,past_key_values=past,use_cache=True,logits_to_keep=1)
   past=out.past_key_values;tokens.append(out.logits[:,-1].argmax(-1));finite.logical_and_(torch.isfinite(out.logits).all());torch.cuda.synchronize();now=time.perf_counter()
   if step==0:ttft=now-start;decode_start=now
   else:times.append(now-previous)
   previous=now
   if step<32:ids=teacher[:,step:step+1];mask=torch.cat((mask,mask.new_ones((len(ids),1))),1)
  actual=torch.stack(tokens).cpu().numpy();ok=bool(finite)
 assert rt.index==1584 and ok
 return dict(TTFT=ttft,TPOT=(now-decode_start)/32,E2E=now-start,decode_wall=now-decode_start,per_step_s=times,argmax_hash=array_hash(actual),finite_logits=ok),actual

def main(a):
 rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
 physical=list(map(int,os.environ['MGO_V2_PHYSICAL_GPUS'].split(',')));assert physical==[0,1,4,5]
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 root=a.output;manifest=json.loads((root/'manifest.json').read_text());model,backing,experts=load_model();pool=None;pinned=None;all_results=[]
 for cell_index,spec in enumerate(manifest):
  out=root/spec['label'];a.output=out;a.inputs=Path(spec['inputs']);a.capacities=spec['capacities'];a.seed=spec['placement_seed'];a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.debug_plan=False
  records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];assert len(records)==spec['batch'] and all(len(x['input_ids'])==spec['context'] for x in records)
  ids=torch.tensor([x['input_ids'] for x in records],device='cuda');batch=len(ids);teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda')
  warm={};warm_tokens={};samples={}
  def setup(policy,phase):
   nonlocal pool,pinned
   a.policy=policy;a.phase=phase
   if pool is None:
    rt=create_selected_runtime(a,model,backing,experts);pool=rt.pinned_expert_pool;pinned=rt.experts
    write(root/f'pinned_store_rank{rank}.json',dict(**rt.pinned_expert_store_receipt,rank=rank,physical_gpu=physical[rank]))
   else:
    for key,value in selected_options().items():setattr(a,key,value)
    a.direct_pinned_source=True;rt=DecodeOffloadRuntime(a,model,backing,pinned)
   assert rt.h2d.direct_pinned and getattr(rt,'graph_executor',None) is None
   rt.stage_frozen_inputs(32)
   from refactor_thread_affinity import configure
   configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
   return rt
  def validate_row(rt,row,tokens,policy):
   validate(rt,row,spec['proofs'][policy],rank)
   return dict(status='PASS',controller=dict(rt.controller.counters),scheduler=dict(rt.h2d.metrics),forward_bytes=rt.transport.forward_bytes,return_bytes=rt.transport.return_bytes,state_hash=array_hash(rt.policy.slots[rank,:rt.cap]),role_hash=array_hash(rt.arena.main_physical[rank]),graph_buffer_bytes=0)
  for policy in POLICIES:
   if rank==0:write(root/'phase.json',dict(stage='WARM_CORRECTNESS',cell=spec['label'],policy=policy))
   rt=setup(policy,'COUNTERS');row,tokens=generation(model,rt,ids,teacher);v=validate_row(rt,row,tokens,policy)
   warm[policy]=row['argmax_hash'];warm_tokens[policy]=tokens.copy();write(out/f'{policy}_warm_rank{rank}.json',dict(**row,validation=v));rt.close();del rt;gc.collect();dist.barrier()
  differences=int(np.count_nonzero(warm_tokens[POLICIES[0]]!=warm_tokens[POLICIES[1]]))
  write(out/f'policy_token_difference_rank{rank}.json',dict(different_tokens=differences,total_tokens=int(warm_tokens[POLICIES[0]].size),scope='BF16 policy reduction order; frozen routes and teacher inputs shared'))
  order=POLICIES if cell_index%2==0 else tuple(reversed(POLICIES))
  for policy in order:
   rt=setup(policy,'MEASURE');gc.collect();torch.cuda.synchronize();dist.barrier()
   from torch._dynamo.utils import counters
   before=dict(counters['stats']);key=f'{spec["label"]}_{policy}_r1';write(root/f'{key}_ready_rank{rank}.json',dict(rank=rank));dist.barrier()
   if rank==0:write(root/'boundary.json',dict(key=key,cell=spec['label'],policy=policy))
   deadline=time.monotonic()+600
   while not (root/f'{key}_GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('primary release boundary')
    time.sleep(.2)
   with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generation(model,rt,ids,teacher)
   v=validate_row(rt,row,tokens,policy);v.update(no_compile=before==dict(counters['stats']),tokens_match_warm=row['argmax_hash']==warm[policy])
   assert v['no_compile'] and v['tokens_match_warm']
   row.update(status='PASS',policy=policy,rank=rank,repeat=1,validation=v,peak_gpu_bytes=torch.cuda.max_memory_allocated(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
   write(out/f'{policy}_r1_measure_rank{rank}.json',row);rt.close();del rt;gc.collect();dist.barrier()
   rows=[json.loads((out/f'{policy}_r1_measure_rank{r}.json').read_text()) for r in range(4)];samples[policy]={m:max(x[m] for x in rows) for m in METRICS}
  if rank==0:
   diffs=[json.loads((out/f'policy_token_difference_rank{r}.json').read_text()) for r in range(4)]
   result=dict(status='PASS',batch=spec['batch'],context=spec['context'],seed=spec['placement_seed'],executor='H0',source='full_pinned',runtime='V3_OPT_PF_OVERLAP',P=2,trigger='T2',decode_steps=32,generated_outputs=33,measured_repeats=1,baseline_policy=POLICIES[0],candidate_policy=POLICIES[1],samples=samples,gains={m:1-samples[POLICIES[1]][m]/samples[POLICIES[0]][m] for m in METRICS},policy_token_differences=diffs,scope='Single-shot rerun on previous BR-adversarial selected seed; prefill policy and cache carry over into decode. No strict layer barriers, diagnostics or recapture. All timings wall-clock; independent max across ranks.')
   write(out/'result.json',result);all_results.append(result);write(root/'phase.json',dict(stage='CELL_COMPLETE',cell=spec['label'],completed_cells=len(all_results)))
  dist.barrier();del ids,teacher;gc.collect()
 if rank==0:write(root/'result.json',dict(status='PASS',results=all_results,primary_policy_runs=len(manifest)*len(POLICIES),scope='Single measurement per requested combination; no repeat stability claim.'))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
