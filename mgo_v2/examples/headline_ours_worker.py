"""Native greedy Near generation with cold expert cache and common release."""
from refactor_measure_worker import *
from mgo_v2.selected_runtime import create_selected_runtime,selected_options
import psutil,hashlib
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')

def begin(rt):
 rt.h2d.synchronize();assert not rt.controller.pending
 rt.event_offset+=rt.index;rt.index=0;rt.mismatch.zero_();rt.valid=None
 assert rt.event_offset%48==0

def generate_live(model,rt,ids,n):
 dist.barrier();torch.cuda.synchronize()
 release=torch.tensor([time.perf_counter_ns()+200_000_000 if dist.get_rank()==0 else 0],device='cuda',dtype=torch.int64)
 dist.broadcast(release,0);start=int(release.item())
 while time.perf_counter_ns()<start:time.sleep(.0001)
 if getattr(rt,'phase_diagnostic',None):rt.phase_diagnostic.start()
 mask=torch.ones_like(ids);past=None;tokens=[];finite=torch.ones((),dtype=torch.bool,device='cuda');stamps=[]
 with torch.inference_mode():
  for step in range(n):
   pos=torch.arange(mask.shape[1]-ids.shape[1],mask.shape[1],device='cuda')[None,:].expand(len(ids),-1)
   out=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=past,use_cache=True,logits_to_keep=1)
   past=out.past_key_values;next_ids=out.logits[:,-1].argmax(-1);tokens.append(next_ids);finite.logical_and_(torch.isfinite(out.logits).all())
   torch.cuda.synchronize();stamps.append(time.perf_counter_ns())
   if step<n-1:ids=next_ids[:,None];mask=torch.cat((mask,mask.new_ones((len(ids),1))),1)
 if getattr(rt,'phase_diagnostic',None):rt.phase_diagnostic.stop()
 actual=torch.stack(tokens,dim=1).cpu().numpy();assert bool(finite) and rt.index==48*n
 return dict(release_ns=start,first_ns=stamps[0],end_ns=stamps[-1],token_ready_ns=stamps,finite_logits=True,output_tokens=n,argmax_hash=array_hash(actual),tokens=actual.tolist())

def validate_state(rt):
 rt.h2d.synchronize();rt.arena.assert_consistent();assert not rt.controller.pending and not rt.mismatch.item()
 assert np.array_equal(rt.keys[rt.arena.main_physical[rt.rank]],rt.policy.slots[rt.rank,:rt.cap])
 state=array_hash(rt.policy.slots);roles=array_hash(np.concatenate(rt.arena.main_physical))
 digest=torch.tensor(list(bytes.fromhex(state+roles)),dtype=torch.uint8,device='cuda');all_digests=[torch.empty_like(digest) for _ in range(4)];dist.all_gather(all_digests,digest)
 assert all(torch.equal(digest,x) for x in all_digests)
 return dict(status='PASS',state_hash=state,role_hash=roles,controller=dict(rt.controller.counters),scheduler=dict(rt.h2d.metrics),main_slots=sum(rt.args.capacities),physical_slots=sum(rt.args.capacities)+rt.world*rt.args.arena_budget)

def main(a):
 assert not (a.capture_eviction_trace and a.record_main_eviction_trace)
 assert not a.capture_eviction_trace or (a.policy=='BR' and a.repeats==1 and not a.smoke and not a.post_generation_diagnostic and not a.decode_layout_fast)
 assert not a.prefill_diagnostic or a.prefill_optimized
 assert not a.prefill_layout_fast or a.prefill_optimized
 assert not a.native_prefill or (a.expert_executor=='native' and a.prefill_optimized)
 rank=int(os.environ['RANK']);assert dist.is_available() and int(os.environ['WORLD_SIZE'])==4
 physical=[0,1,4,5];assert os.environ['MGO_V2_PHYSICAL_GPUS']=='0,1,4,5'
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 spec=next(x for x in json.loads(Path(os.environ.get('MGO_HEADLINE_WORKLOADS',str(ROOT/'WORKLOADS.json'))).read_text())['cells'] if x['cell']==a.cell)
 a.local_batch=1 if a.smoke else spec['local_batch'];a.capacities=[x-(0 if a.capture_eviction_trace else 2) for x in spec.get('expert_slots_per_rank',[461,461,461,460])];a.seed=42;a.phase='COUNTERS';a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.debug_plan=False;a.live_routes=True
 model,backing,experts=load_model();rt=create_selected_runtime(a,model,backing,experts,arm='V1_OPT_NOPF_BARRIER' if a.capture_eviction_trace else None)
 assert rt.cache.numel()*rt.cache.element_size()==(a.capacities[rank]+a.arena_budget)*EB
 from refactor_thread_affinity import configure
 configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
 write(a.output/f'pinned_rank{rank}.json',rt.pinned_expert_store_receipt)
 # Match normal cache tensor dispatch keys and both storage-offset classes.
 # Synthetic inference tensors do not cover views of the persistent arena.
 for row_count in [1,2]:
  for slot in [0,1]:
   with torch.inference_mode():
    w=rt.cache[slot];w.zero_()
    rt.kernel(torch.zeros((row_count,2048),device='cuda',dtype=torch.bfloat16),w[:1572864].view(768,2048),w[1572864:3145728].view(768,2048),w[3145728:].view(2048,768))
 torch.cuda.synchronize()
 def reset_live():
  rt.reset();rt.event_offset=0
  from mgo_v2.eviction import GateHistory
  from mgo_v2.live_metadata import LiveMetadata
  rt.gate_history=GateHistory(48,128,128);rt.metadata=LiveMetadata(a.local_batch,rt.gate_history)
  configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
 if a.prefill_optimized and not a.prefill_diagnostic and not a.capture_eviction_trace:
  if rank==0:write(a.output/'phase.json',dict(phase='prefill_validation',cell=a.cell))
  rows=json.loads(Path(spec['warmup']['path']).read_text())['requests']
  local=rows[rank:rank+1] if a.smoke else rows[rank*a.local_batch:(rank+1)*a.local_batch]
  ids=torch.tensor([r['input_ids'][-32:] if a.smoke else r['input_ids'] for r in local],device='cuda')
  a.validate_prefill_optimized=True;begin(rt)
  check=generate_live(model,rt,ids,1);state=validate_state(rt)
  assert rt.prefill_metadata_checks==48 and len(rt.prefill_numerics)==48
  assert not a.prefill_layout_fast or rt.prefill_layout_checks==48
  assert rt.transport.calls==48*3
  write(a.output/f'prefill_validation_rank{rank}.json',dict(status='PASS',metadata_exact_checks=rt.prefill_metadata_checks,layout_exact_checks=getattr(rt,'prefill_layout_checks',0),per_layer_numerics=rt.prefill_numerics,state=state,output=check,semantics='Exact metadata and same-part expert-order versus BF16 rank-partial numerical comparison; not bitwise parity'))
  a.validate_prefill_optimized=False;reset_live()
 n=257 if a.capture_eviction_trace else (1 if a.prefill_diagnostic else (2 if a.smoke else 64));repeats=1 if a.prefill_diagnostic or a.smoke else a.repeats
 for repeat in ((1,) if a.capture_eviction_trace else range(repeats+1)):
  phase='warmup' if repeat==0 else 'target';rows=json.loads(Path(spec[phase]['path']).read_text())['requests']
  local=rows[rank:rank+1] if a.smoke else rows[rank*a.local_batch:(rank+1)*a.local_batch]
  ids=torch.tensor([r['input_ids'][-32:] if a.smoke else r['input_ids'] for r in local],device='cuda')
  # Owner amendment: retain compiled code/full pinned source, clear expert state.
  if repeat:
   rt.reset();rt.event_offset=0
   from mgo_v2.eviction import GateHistory
   from mgo_v2.live_metadata import LiveMetadata
   rt.gate_history=GateHistory(48,128,128);rt.metadata=LiveMetadata(a.local_batch,rt.gate_history)
   configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
  a.validate_decode_layout=a.decode_layout_fast and repeat==0
  rt.record_main_eviction_trace=bool(a.record_main_eviction_trace and repeat)
  if rt.record_main_eviction_trace:
   rt.main_eviction_trace_histograms=[];rt.main_eviction_trace_gates=[];rt.main_eviction_trace_events=[]
  begin(rt);a.phase='COUNTERS' if repeat==0 else 'MEASURE';cache_before=validate_state(rt);calls=rt.transport.calls
  assert np.all(rt.keys<0) and not rt.controller.pending,'expert cache not cold'
  if rank==0:write(a.output/'phase.json',dict(system='Ours',cell=a.cell,phase=phase,repeat=repeat,smoke=a.smoke))
  gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  from torch._dynamo.utils import counters
  before=dict(counters['stats'])
  if repeat and a.prefill_diagnostic:
   from prefill_phase_diagnostics import PrefillDiagnostics
   rt.phase_diagnostic=PrefillDiagnostics(rt);rt.phase_diagnostic.install(model)
  if a.capture_eviction_trace:
   from main_eviction_trace import RoutingCapture
   capture=RoutingCapture(rt,n*48);capture.install()
   result=generate_live(model,rt,ids,n)
   capture.finish(a.output)
  elif repeat:
   with torch._dynamo.config.patch(error_on_recompile=True):result=generate_live(model,rt,ids,n)
  else:result=generate_live(model,rt,ids,n)
  if repeat and a.prefill_diagnostic:
   diagnostic=rt.phase_diagnostic.finish((result['end_ns']-result['release_ns'])/1e9)
   write(a.output/f'diagnostic_rank{rank}.json',diagnostic);rt.phase_diagnostic=None
  if a.validate_decode_layout:
   assert rt.decode_layout_checks==(n-1)*48
   write(a.output/f'decode_layout_validation_rank{rank}.json',dict(status='PASS',exact_checks=rt.decode_layout_checks,scope='all warmup decode layers; integer indices and physical slot parity'))
  if rt.record_main_eviction_trace:
   hist=np.stack(rt.main_eviction_trace_histograms).astype(np.int32)
   gates=np.stack(rt.main_eviction_trace_gates).astype(np.float32)
   events=np.asarray(rt.main_eviction_trace_events,dtype=np.int32)
   assert hist.shape==(48*n,4,128) and gates.shape==(48*n,128)
   assert np.array_equal(events,np.arange(48*n,dtype=np.int32))
   raw=hist.tobytes()+gates.tobytes()+events.tobytes()
   digest=np.frombuffer(hashlib.sha256(raw).digest(),np.uint8).copy()
   d=torch.tensor(digest,device='cuda');all_d=torch.empty((4,32),dtype=torch.uint8,device='cuda')
   dist.all_gather_into_tensor(all_d.view(-1),d);assert bool((all_d==all_d[0]).all())
   if rank==0:
    np.savez_compressed(a.output/'main_eviction_trace.npz',histograms=hist,gates=gates,events=events)
    write(a.output/'main_eviction_trace_meta.json',dict(
     status='PASS',cell=a.cell,source_policy=a.policy,local_batch=a.local_batch,global_requests=4*a.local_batch,
     input_tokens=int(ids.shape[1]),output_tokens=n,events=int(len(events)),prefill_events=48,
     decode_events=int((n-1)*48),main_capacities=list(map(int,a.capacities)),
     captured_runtime_prefetch_P=int(a.arena_budget),captured_runtime_trigger=a.trigger,
     simulator_scope='MAIN-only P0 eviction replay; frozen raw demand/gate stream',
     trace_sha256=hashlib.sha256(raw).hexdigest()))
   rt.record_main_eviction_trace=False
  validation=validate_state(rt);assert rt.transport.calls-calls==(n-1+int(a.prefill_optimized))*48*2
  no_compile=before==dict(counters['stats']);assert a.capture_eviction_trace or not repeat or no_compile
  result.update(expert_executor=a.expert_executor,native_prefill=a.native_prefill,decode_layout_fast=a.decode_layout_fast,prefill_layout_fast=a.prefill_layout_fast,prefill_optimized=a.prefill_optimized,expert_cache_start='empty',system='Ours',policy=a.policy,rank=rank,physical_gpu=physical[rank],repeat=repeat,phase=phase,smoke=a.smoke,validation=validation,cache_before=cache_before,no_compile=no_compile,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),host_rss_bytes=psutil.Process().memory_info().rss,pinned_host_bytes=rt.pinned_expert_store_receipt['bytes'],request_ids=[r['request_id'] for r in local])
  write(a.output/f'repeat{repeat}_rank{rank}.json',result);dist.barrier()
  if rank==0:
   rr=[json.loads((a.output/f'repeat{repeat}_rank{r}.json').read_text()) for r in range(4)];assert len(set(x['release_ns'] for x in rr))==1
   release=rr[0]['release_ns'];first=max(x['first_ns'] for x in rr);end=max(x['end_ns'] for x in rr)
   row=dict(status='PASS',expert_executor=a.expert_executor,repeat=repeat,TTFT=(first-release)/1e9,TPOT=(end-first)/1e9/(n-1) if n>1 else None,E2E=(end-release)/1e9,throughput=4*a.local_batch*n/((end-release)/1e9),output_tokens=n,global_requests=4*a.local_batch,smoke=a.smoke)
   write(a.output/f'repeat{repeat}.json',row);print(json.dumps(row),flush=True)
  dist.barrier()
 if a.post_generation_diagnostic:
  from generation_phase_diagnostics import GenerationDiagnostics
  expected_tokens=result['tokens'];expected_state=validation
  reset_live();begin(rt);a.phase='MEASURE'
  if rank==0:write(a.output/'phase.json',dict(phase='post_generation_diagnostic',cell=a.cell,policy=a.policy))
  rt.phase_diagnostic=GenerationDiagnostics(rt);rt.phase_diagnostic.install(model)
  with torch._dynamo.config.patch(error_on_recompile=True):check=generate_live(model,rt,ids,n)
  diagnostic=rt.phase_diagnostic.finish((check['end_ns']-check['release_ns'])/1e9);rt.phase_diagnostic=None
  state=validate_state(rt)
  assert check['tokens']==expected_tokens and state['state_hash']==expected_state['state_hash'] and state['role_hash']==expected_state['role_hash']
  diagnostic.update(output=check,validation=state,token_and_cache_parity=True)
  write(a.output/f'generation_diagnostic_rank{rank}.json',diagnostic)
 if a.post_prefill_diagnostic:
  assert a.prefill_layout_fast and not a.prefill_diagnostic
  expected_first=[t[0] for t in result['tokens']]
  reset_live();begin(rt);a.phase='MEASURE'
  if rank==0:write(a.output/'phase.json',dict(phase='post_prefill_diagnostic',cell=a.cell))
  from prefill_phase_diagnostics import PrefillDiagnostics
  rt.phase_diagnostic=PrefillDiagnostics(rt);rt.phase_diagnostic.install(model)
  from torch._dynamo.utils import counters
  before=dict(counters['stats'])
  with torch._dynamo.config.patch(error_on_recompile=True):check=generate_live(model,rt,ids,1)
  diagnostic=rt.phase_diagnostic.finish((check['end_ns']-check['release_ns'])/1e9);rt.phase_diagnostic=None
  assert before==dict(counters['stats']) and [t[0] for t in check['tokens']]==expected_first
  assert rt.transport.calls==96
  diagnostic.update(validation=validate_state(rt),first_token_parity=True,no_compile=True)
  write(a.output/f'post_diagnostic_rank{rank}.json',diagnostic)
 rt.close()
 if rank==0:write(a.output/'result.json',dict(status='PASS',system='Ours',expert_executor=a.expert_executor,policy=a.policy,cell=a.cell,smoke=a.smoke,primary_repeats=repeats,prefill_optimized=a.prefill_optimized,headline_eligible=not (a.prefill_diagnostic or a.capture_eviction_trace or a.record_main_eviction_trace)))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--expert-executor',choices=('h0','native'),default='h0');p.add_argument('--native-prefill',action='store_true');p.add_argument('--record-main-eviction-trace',action='store_true');p.add_argument('--capture-eviction-trace',action='store_true');p.add_argument('--policy',choices=('BR','LA_CA_NEAR'),default='LA_CA_NEAR');p.add_argument('--post-generation-diagnostic',action='store_true');p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--repeats',type=int,choices=range(1,6),default=3);p.add_argument('--prefill-optimized',action='store_true');p.add_argument('--prefill-diagnostic',action='store_true');p.add_argument('--prefill-layout-fast',action='store_true');p.add_argument('--post-prefill-diagnostic',action='store_true');g=p.add_mutually_exclusive_group();g.add_argument('--decode-layout-fast',dest='decode_layout_fast',action='store_true');g.add_argument('--legacy-decode-layout',dest='decode_layout_fast',action='store_false');p.set_defaults(decode_layout_fast=False);main(p.parse_args())
