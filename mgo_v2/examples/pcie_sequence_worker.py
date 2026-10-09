"""Counter-ordered primary cohorts; reuse only code, model and pinned source."""
from headline_ours_worker import *
from pcie_host import affinity
from run_pcie_ours_job import ARMS
from pcie_host import configure_pcie as configure
from mgo_v2.eviction import GateHistory
from mgo_v2.live_metadata import LiveMetadata


def main(a):
 rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
 assert os.environ['MGO_V2_PHYSICAL_GPUS']=='0,1,4,5'
 cpus=affinity(rank)
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85)
 torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 a.pcie_cpu_group=dist.new_group(backend='gloo')
 spec=next(x for x in json.loads(Path(os.environ['MGO_HEADLINE_WORKLOADS']).read_text())['cells'] if x['cell']==a.cell)
 assert spec['global_requests']==64 and spec['local_batch']==16 and spec['input_tokens']==512
 a.local_batch=16;a.capacities=[459,459,459,458];a.seed=42;a.phase='COUNTERS';a.comm_mode='current'
 a.staging_cpu_team=cpus[1:3];a.debug_plan=True;a.live_routes=True;a.pcie_native_controller=True
 a.pcie_phase_diagnostic=False;a.pcie_g2g_first_serial=True;a.prefetch_off=True
 a.prefill_optimized=True;a.prefill_layout_fast=True;a.decode_layout_fast=True;a.native_prefill=True
 a.expert_executor='native';a.capture_eviction_trace=False;a.record_main_eviction_trace=False
 a.policy='LA_CA_NEAR';a.pcie_quota_mode='rank_order';a.pcie_peer_costs=None
 a.prefill_diagnostic=False;a.post_generation_diagnostic=False;a.post_prefill_diagnostic=False
 cohort=a.output
 arms=['R-NEAR','G-NEAR'] if a.sequence=='stage1' else ['G-BR','G-CA','G-NUMA-CA','G-NEAR','R-BR','R-CA']
 paths={arm:cohort/arm for arm in arms}
 for path in paths.values():path.mkdir(exist_ok=True)
 costs=np.asarray(json.loads((PKG/'experiments/pcie_topology_ablation_20261009/microbench_physical_cores/peer_costs.json').read_text())['matrix'],np.int64)
 model,backing,experts=load_model();rt=create_selected_runtime(a,model,backing,experts)
 assert sum(a.capacities)+4*a.arena_budget==1843 and a.arena_budget==2
 assert rt.cache.numel()*rt.cache.element_size()==(a.capacities[rank]+2)*EB
 for row_count in (1,2):
  for slot in (0,1):
   with torch.inference_mode():
    w=rt.cache[slot];w.zero_();rt.kernel(torch.zeros((row_count,2048),device='cuda',dtype=torch.bfloat16),w[:1572864].view(768,2048),w[1572864:3145728].view(768,2048),w[3145728:].view(2048,768))
 torch.cuda.synchronize()
 def reset(arm):
  a.policy,a.pcie_quota_mode=ARMS[arm];a.pcie_peer_costs=costs if arm=='G-NUMA-CA' else None
  a.output=paths[arm];rt.reset();rt.event_offset=0
  rt.decode_layout_checks=0
  rt.gate_history=GateHistory(48,128,128);rt.metadata=LiveMetadata(16,rt.gate_history)
  receipt=configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
  write(cohort/f'affinity_rank{rank}.json',receipt)
 def input_ids(phase):
  rows=json.loads(Path(spec[phase]['path']).read_text())['requests'][rank*16:(rank+1)*16]
  return rows,torch.tensor([r['input_ids'] for r in rows],device='cuda')
 # Same one-token prefill checks as the headline harness, outside every sample.
 reset(arms[0]);rows,ids=input_ids('warmup');a.validate_prefill_optimized=True;begin(rt)
 checked=generate_live(model,rt,ids,1);state=validate_state(rt)
 assert rt.prefill_metadata_checks==48 and rt.prefill_layout_checks==48 and len(rt.prefill_numerics)==48
 assert rt.transport.calls==144
 write(cohort/f'prefill_validation_rank{rank}.json',dict(status='PASS',output=checked,state=state,per_layer_numerics=rt.prefill_numerics))
 a.validate_prefill_optimized=False
 previous={}
 def generation(arm,repeat,diagnostic=False):
  a.pcie_phase_diagnostic=diagnostic;a.pcie_capture_decisions=diagnostic and arm=='G-NEAR'
  reset(arm);a.phase='COUNTERS' if repeat==0 else 'MEASURE';a.validate_decode_layout=repeat==0
  phase='warmup' if repeat==0 else 'target';rows,ids=input_ids(phase);begin(rt)
  empty=validate_state(rt);assert np.all(rt.keys<0) and not rt.controller.pending
  write(a.output/f'pinned_rank{rank}.json',rt.pinned_expert_store_receipt)
  if rank==0:
   stamp=dict(phase=phase,arm=arm,repeat=repeat,generation_started_unix=time.time())
   write(cohort/'phase.json',stamp);write(a.output/'phase.json',stamp)
  gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  from torch._dynamo.utils import counters
  before=dict(counters['stats'])
  if repeat:
   with torch._dynamo.config.patch(error_on_recompile=True):result=generate_live(model,rt,ids,64)
  else:result=generate_live(model,rt,ids,64)
  state=validate_state(rt);assert rt.transport.calls==6144 and rt.index==3072 and rt.debug_plan_checks==3072
  assert not repeat or before==dict(counters['stats'])
  assert state['physical_slots']==1843 and state['scheduler']['background_copies']==0
  if repeat:
   expected=previous.get(arm)
   if expected is not None:assert expected==(result['argmax_hash'],state['state_hash'])
   previous[arm]=(result['argmax_hash'],state['state_hash'])
  trace=rt.policy.native_pcie.trace();assert trace.shape==(3072,61)
  quotas=np.column_stack((trace[:,19],trace[:,48:52])).astype(np.int64)
  assert np.all(quotas[:,0]==quotas[:,1:].sum(axis=1))
  if a.pcie_quota_mode=='group_balanced':assert np.all(np.abs(quotas[:,1:3].sum(axis=1)-quotas[:,3:5].sum(axis=1))<=1)
  np.save(a.output/f'policy_trace_repeat{repeat}_rank{rank}.npy',trace)
  np.save(a.output/f'quota_repeat{repeat}_rank{rank}.npy',quotas)
  if diagnostic:
   write(a.output/f'phases_repeat{repeat}_rank{rank}.json',rt.pcie_phase_trace)
   if rt.pcie_decision_capture is not None:rt.pcie_decision_capture.save(a.output/f'decision_capture_repeat{repeat}.npz')
  if repeat==0:assert rt.decode_layout_checks==3024
  result.update(arm=arm,policy=a.policy,quota_mode=a.pcie_quota_mode,rank=rank,physical_gpu=[0,1,4,5][rank],repeat=repeat,phase=phase,
                system='Ours',smoke=False,profiled=diagnostic,expert_executor='native',expert_cache_start='empty',native_prefill=True,prefetch_off=True,
                pcie_native_controller=True,pcie_g2g_first_serial=True,debug_plan_checks=rt.debug_plan_checks,validation=state,cache_before=empty,
                no_compile=before==dict(counters['stats']),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                host_rss_bytes=psutil.Process().memory_info().rss,pinned_host_bytes=rt.pinned_expert_store_receipt['bytes'],
                request_ids=[r['request_id'] for r in rows],forward_wire_bytes=rt.transport.forward_bytes,return_wire_bytes=rt.transport.return_bytes)
  write(a.output/f'repeat{repeat}_rank{rank}.json',result);dist.barrier(group=a.pcie_cpu_group)
  if rank==0:
   rr=[json.loads((a.output/f'repeat{repeat}_rank{r}.json').read_text()) for r in range(4)]
   assert len({r['release_ns'] for r in rr})==1
   release=rr[0]['release_ns'];first=max(r['first_ns'] for r in rr);end=max(r['end_ns'] for r in rr)
   row=dict(status='PASS',arm=arm,repeat=repeat,TTFT=(first-release)/1e9,TPOT=(end-first)/1e9/63,E2E=(end-release)/1e9,
            throughput=4096/((end-release)/1e9),output_tokens=64,global_requests=64,smoke=False,profiled=diagnostic)
   write(a.output/f'repeat{repeat}.json',row);print(json.dumps(row),flush=True)
  dist.barrier(group=a.pcie_cpu_group)
 for arm in arms:generation(arm,0)
 schedules=[arms,arms[::-1],arms]
 for repeat,order in enumerate(schedules,1):
  for arm in order:generation(arm,repeat)
 extra=torch.zeros(1,dtype=torch.int64)
 if rank==0:
  spread={}
  for arm in arms:
   samples=[json.loads((paths[arm]/f'repeat{r}.json').read_text())['TPOT'] for r in (1,2,3)]
   spread[arm]=(max(samples)-min(samples))/np.median(samples)
  extra[0]=int(any(v>.05 for v in spread.values()))
  write(cohort/'stability_decision.json',dict(threshold=.05,relative_tpot_spreads=spread,additional_repeats=2 if extra[0] else 0,scope='all arms in this cohort'))
 dist.broadcast(extra,0,group=a.pcie_cpu_group)
 if extra.item():
  for repeat,order in ((4,arms[::-1]),(5,arms)):
   for arm in order:generation(arm,repeat)
 repeats=5 if extra.item() else 3
 for arm in arms[::-1]:generation(arm,-1,diagnostic=True)
 if rank==0:
  for arm in arms:
   samples=[json.loads((paths[arm]/f'repeat{r}.json').read_text()) for r in range(1,repeats+1)]
   statistics={}
   for metric in ('TTFT','TPOT','E2E','throughput'):
    values=np.array([s[metric] for s in samples]);statistics[metric]=dict(median=float(np.median(values)),mean=float(values.mean()),sd=float(values.std(ddof=1)),min=float(values.min()),max=float(values.max()),values=values.tolist())
   write(paths[arm]/'result.json',dict(status='PASS',system='Ours',arm=arm,cell=a.cell,primary_repeats=repeats,profiled=False,
                                    statistics=statistics,schedule=[list(x) for x in schedules],source='numa_shared_full_pinned',controller='native C++'))
  write(cohort/'result.json',dict(status='PASS',sequence=a.sequence,arms=arms,primary_repeats=repeats,profiled=False))
 dist.barrier(group=a.pcie_cpu_group);rt.close();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--sequence',choices=('stage1','stage2'),required=True);p.add_argument('--numa-shared-source-root',required=True)
 # Shared launcher argv also describes the frozen runtime explicitly.
 p.add_argument('--policy');p.add_argument('--expert-executor');p.add_argument('--pcie-quota-mode');p.add_argument('--repeats',type=int)
 for flag in ('native-prefill','prefetch-off','prefill-optimized','prefill-layout-fast','decode-layout-fast','pcie-native-controller','pcie-g2g-first-serial'):
  p.add_argument('--'+flag,action='store_true')
 main(p.parse_args())
