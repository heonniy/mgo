"""R4 full-model A/B: current staged H2D vs one 54-GiB pinned expert copy per rank.

Reuses the established C30 / local-B128 / decode64 / H1b optimized runtime and
the same frozen BR workload.  The only intended difference is expert source:
  STAGED      : file-backed CPU expert -> two pinned staging buffers -> GPU
  FULL_PINNED : rank-private full pinned expert copy -> GPU

Physical GPUs are 0,1,4,5 only.  Full pinned allocation/copy is untimed.
"""
from refactor_measure_worker import *
from physical_repeat_rule import final_unstable
from mgo_v2.graph_expert import GraphExpertExecutor
from mgo_v2.selected_runtime import create_explicit_runtime
from mgo_v2.pinned_h2d import build_full_pinned_expert_store
import statistics,resource

MODES=('STAGED','FULL_PINNED')
PINNED_EXPECTED=48*128*EB

def host_available():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
 raise RuntimeError('MemAvailable unavailable')

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
 physicals=[int(x) for x in os.environ['MGO_V2_PHYSICAL_GPUS'].split(',')];assert physicals==[0,1,4,5]
 physical=physicals[rank];a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical)]
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85)
 torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))

 source=json.loads((a.inputs/'receipt.json').read_text())
 a.seed=source['winner']['placement_seed'];a.capacities=source['capacities'];assert len(a.capacities)==4
 a.comm_mode='current';a.phase='COUNTERS';a.policy='BR';a.debug_plan=False
 case=dict(label='BR_H1b',policy='BR',executor='H1b',P=2,trigger='T2',horizon=64,
  overlap=True,partial_precision='bf16',runtime_arm='V3_OPT_PF_OVERLAP',
  staging_backend='torch',unique_combine=True,async_metadata_inputs=True,
  isolated_cpu_threads=True,fixed_staging_team=True,post_expert_barrier=False)
 a.staging_cpu_team=cpus[1:3];a.direct_pinned_source=False

 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records);assert batch==128
 model,backing,staged_experts=load_model()
 length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda')
 mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
 teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda')
 horizon=64;proof=json.loads((a.inputs/'BR_P2_proof.json').read_text());assert proof['horizon']==horizon

 rt=create_explicit_runtime(a,model,backing,staged_experts,case);rt.stage_frozen_inputs(horizon)
 rt.policy_kind=0;rt.proof=source['proofs']['BR'];rt.reference=json.loads((a.inputs/'BR_fetches.json').read_text())

 # Reuse the committed H1b workload signatures; graph construction is still
 # local to this process but no signature-discovery full run is needed.
 graph=GraphExpertExecutor(rt.cache,rt.kernel,wrapper=True)
 prior=Path('/home/hwlee/mgo-results/critical_path_admission_followup_20261006/b3/C30/B3_C30_CLEAN_B128_H64')
 saved=json.loads((prior/f'graph_signatures_rank{rank}.json').read_text())
 graph.signatures=set(map(tuple,saved['signatures']))
 rt.h2d.close()
 graph.build(lambda row:write(a.output/f'graph_progress_rank{rank}.json',row))
 write(a.output/f'graph_receipt_rank{rank}.json',graph.receipt())

 # One private 54-GiB pinned expert copy per rank.  Keep the original mapped
 # sources alive so STAGED can be measured in the same process.
 dist.barrier()
 before=host_available()
 if before < 320*2**30:raise RuntimeError(f'need >=320 GiB host headroom before full pinned allocation, have {before/2**30:.1f} GiB')
 t0=time.perf_counter()
 pinned_pool,pinned_experts=build_full_pinned_expert_store(staged_experts,EB)
 pinned_init_s=time.perf_counter()-t0
 assert pinned_pool.numel()*pinned_pool.element_size()==PINNED_EXPECTED
 assert pinned_pool.is_pinned()
 _=float(pinned_pool[:,::4096].float().sum())  # first-touch before timing
 dist.barrier()
 write(a.output/f'pinned_store_rank{rank}.json',dict(status='PASS',rank=rank,physical_gpu=physical,
  bytes=PINNED_EXPECTED,gib=PINNED_EXPECTED/2**30,init_seconds=pinned_init_s,
  host_available_before=before,host_available_after=host_available()))

 def configure(mode,phase):
  assert mode in MODES
  a.phase=phase;a.policy='BR';a.direct_pinned_source=(mode=='FULL_PINNED')
  rt.experts=pinned_experts if a.direct_pinned_source else staged_experts
  rt.policy_kind=0;rt.proof=source['proofs']['BR'];rt.reference=json.loads((a.inputs/'BR_fetches.json').read_text())
  rt.reset();rt.graph_executor=graph;rt.graph_check=False
  from refactor_thread_affinity import configure as configure_affinity
  rt.measurement_thread_placement=configure_affinity(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)

 # Correctness/counter parity on both source paths.
 warm_tokens=None;reference=None
 for mode in MODES:
  configure(mode,'COUNTERS')
  row,tokens=generate(model,rt,ids,mask,teacher,horizon);validate(rt,row,proof,rank)
  observed=dict(counters=dict(rt.controller.counters),copies=rt.h2d.metrics['copies'],
   bytes=rt.h2d.metrics['bytes'],forward=rt.transport.forward_bytes,back=rt.transport.return_bytes)
  if warm_tokens is None:
   warm_tokens=tokens.copy();reference=observed
  else:
   assert np.array_equal(tokens,warm_tokens),'full-pinned argmax differs from staged'
   assert observed==reference,'full-pinned physical/controller counters differ'
  write(a.output/f'{mode}_correctness_rank{rank}.json',dict(status='PASS',mode=mode,
   argmax_hash=row['argmax_hash'],observed=observed,scheduler_metrics=dict(rt.h2d.metrics),
   direct_pinned=bool(rt.h2d.direct_pinned),pinned_store_bytes=PINNED_EXPECTED))

 from torch._dynamo.utils import counters
 samples={m:[] for m in MODES}
 for repeat in (1,2):
  order=list(MODES) if repeat==1 else list(reversed(MODES))
  for mode in order:
   configure(mode,'MEASURE');torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
   before_compile=dict(counters['stats']);label=f'{mode}_r{repeat}'
   write(a.output/f'{label}_ready_rank{rank}.json',dict(rank=rank,mode=mode,repeat=repeat));dist.barrier()
   if rank==0:write(a.output/'boundary.json',dict(key=label,mode=mode,repeat=repeat,order=order))
   deadline=time.monotonic()+600
   while not (a.output/f'{label}_GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('full pinned A/B boundary release')
    time.sleep(.2)
   usage_before=resource.getrusage(resource.RUSAGE_SELF)
   with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,horizon)
   usage_after=resource.getrusage(resource.RUSAGE_SELF)
   assert before_compile==dict(counters['stats']) and np.array_equal(tokens,warm_tokens)
   validate(rt,row,proof,rank)
   observed=dict(counters=dict(rt.controller.counters),copies=rt.h2d.metrics['copies'],
    bytes=rt.h2d.metrics['bytes'],forward=rt.transport.forward_bytes,back=rt.transport.return_bytes)
   assert observed==reference
   row.update(TTFT=row['E2E_wall']-row['decode_wall'],status='PASS',rank=rank,mode=mode,repeat=repeat,
    direct_pinned=bool(rt.h2d.direct_pinned),pinned_store_bytes=PINNED_EXPECTED,
    no_compile_in_measure=True,scheduler_metrics=dict(rt.h2d.metrics),
    peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    process_user_s=usage_after.ru_utime-usage_before.ru_utime,
    process_system_s=usage_after.ru_stime-usage_before.ru_stime)
   write(a.output/f'{label}_measure_rank{rank}.json',row);dist.barrier()
   critical={k:max(json.loads((a.output/f'{label}_measure_rank{r}.json').read_text())[k] for r in range(world))
             for k in ['TTFT','E2E_wall','decode_wall','TPOT']}
   samples[mode].append(critical)
   if rank==0:write(a.output/'phase.json',dict(stage='MEASURE_COMPLETE',mode=mode,repeat=repeat,samples=samples))

 if rank==0:
  estimates={m:{k:statistics.median(x[k] for x in rows) for k in ['TTFT','E2E_wall','decode_wall','TPOT']}
             for m,rows in samples.items()}
  gains={k:1-estimates['FULL_PINNED'][k]/estimates['STAGED'][k] for k in estimates['STAGED']}
  spreads={m:{k:(max(x[k] for x in rows)-min(x[k] for x in rows))/statistics.mean(x[k] for x in rows)
              for k in ['TTFT','E2E_wall','decode_wall','TPOT']} for m,rows in samples.items()}
  write(a.output/'result.json',dict(status='PASS',scope='R4 C30 B128 decode64 H1b BR staged-vs-rank-private-full-pinned',
   physical_gpus=[0,1,4,5],pinned_bytes_per_rank=PINNED_EXPECTED,samples=samples,
   estimates=estimates,gains=gains,relative_spread=spreads,
   caution='Full pinned store allocation/copy is outside timing. Two repeats only; this is an implementation A/B before main-table runs.'))

 rt.close();dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
