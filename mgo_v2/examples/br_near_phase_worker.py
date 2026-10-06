"""One B64/L512 paired decode diagnostic, using completed primary inputs."""
from r4_br_near_h0_worker import *
from br_near_phase_probe import PhaseProbe
import gzip

def main(a):
 rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
 physical=list(map(int,os.environ['MGO_V2_PHYSICAL_GPUS'].split(',')));assert physical==[0,1,4,5]
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 root=a.output;source=Path('/home/hwlee/mgo-results/r4_br_near_h0_20261007')
 spec=next(x for x in json.loads((source/'manifest.json').read_text()) if x['label']=='B64_L512');write(root/f'spec_rank{rank}.json',spec)
 a.inputs=Path(spec['inputs']);a.capacities=spec['capacities'];a.seed=spec['placement_seed'];a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.debug_plan=False
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank]
 ids=torch.tensor([x['input_ids'] for x in records],device='cuda');teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*64:(rank+1)*64],device='cuda')
 model,backing,experts=load_model();pool=None;pinned=None;results={};tokens_by_policy={}
 def setup(policy,phase):
  nonlocal pool,pinned
  a.policy=policy;a.phase=phase
  if pool is None:
   rt=create_selected_runtime(a,model,backing,experts);pool=rt.pinned_expert_pool;pinned=rt.experts
   write(root/f'pinned_store_rank{rank}.json',rt.pinned_expert_store_receipt)
  else:
   for key,value in selected_options().items():setattr(a,key,value)
   a.direct_pinned_source=True;rt=DecodeOffloadRuntime(a,model,backing,pinned)
  rt.stage_frozen_inputs(32)
  from refactor_thread_affinity import configure
  configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
  return rt
 warm={}
 for policy in POLICIES:
  if rank==0:write(root/'phase.json',dict(stage='WARM_CORRECTNESS',policy=policy))
  rt=setup(policy,'COUNTERS');row,tokens=generation(model,rt,ids,teacher);validate(rt,row,spec['proofs'][policy],rank)
  warm[policy]=row['argmax_hash'];write(root/f'{policy}_warm_rank{rank}.json',row);rt.close();del rt;gc.collect();dist.barrier()
 for policy in POLICIES:
  rt=setup(policy,'MEASURE');probe=PhaseProbe(rt)
  from refactor_thread_affinity import configure
  configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
  gc.collect();torch.cuda.synchronize();dist.barrier()
  if rank==0:write(root/'phase.json',dict(stage='DIAGNOSTIC',policy=policy))
  from torch._dynamo.utils import counters
  before=dict(counters['stats'])
  with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generation(model,rt,ids,teacher)
  no_compile=before==dict(counters['stats']);raw=probe.finish();validate(rt,row,spec['proofs'][policy],rank)
  primary=json.loads((source/'B64_L512'/f'{policy}_r1_measure_rank{rank}.json').read_text())
  assert no_compile and row['argmax_hash']==warm[policy]==primary['argmax_hash']
  assert len(raw['work'])==1536 and all(sum(s['name']==name for s in raw['spans'])==1536 for name in ['moe','compute','metadata','forward','return'])
  row.update(status='PASS',rank=rank,physical_gpu=physical[rank],policy=policy,no_compile=no_compile,tokens_match_primary=True,scheduler=dict(rt.h2d.metrics),controller=dict(rt.controller.counters),forward_bytes=rt.transport.forward_bytes,return_bytes=rt.transport.return_bytes,primary_TPOT=primary['TPOT'],peak_gpu_bytes=torch.cuda.max_memory_allocated())
  with gzip.open(root/f'{policy}_trace_rank{rank}.json.gz','wt') as f:json.dump(raw,f,separators=(',',':'))
  write(root/f'{policy}_diagnostic_rank{rank}.json',row);results[policy]=row;tokens_by_policy[policy]=tokens
  rt.close();del rt,probe,raw;gc.collect();dist.barrier()
 write(root/f'policy_difference_rank{rank}.json',dict(different_tokens=int(np.count_nonzero(tokens_by_policy['BR']!=tokens_by_policy['LA_CA_NEAR'])),total_tokens=int(tokens.size)))
 if rank==0:write(root/'result.json',dict(status='PASS',cell='B64_L512',policies=list(POLICIES),diagnostic_runs=2,scope='CUDA stream spans; instrumented, not primary timing.'))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
