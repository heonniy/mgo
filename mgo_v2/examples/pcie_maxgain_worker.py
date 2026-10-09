"""Real best64 search: cold-cache nomination, full64 screen, final repetitions.

Selection and corpus preparation happen outside this GPU worker. Both placements
use exactly the same grouped executor and native metadata implementation.
"""
from pcie_sequence_worker import *
from mgo_v2.pcie_actual_fetch_capture import ActualFetchCapture


class CurrentRoutingCapture:
 def __init__(self,rt):
  self.rt=rt;self.inputs=[];self.original=rt.controller.plan_current
  def call(event,selected,weights,origins,gate):
   assert event==len(self.inputs)
   self.inputs.append(tuple(np.array(x,copy=True) for x in (selected,weights,origins,gate)))
   return self.original(event,selected,weights,origins,gate)
  rt.controller.plan_current=call
 def finish(self,path):
  self.rt.controller.plan_current=self.original
  offsets=np.r_[0,np.cumsum([len(x[0]) for x in self.inputs])].astype(np.int64)
  arrays=dict(events=np.arange(len(self.inputs),dtype=np.int64),offsets=offsets,
              selected=np.concatenate([x[0] for x in self.inputs]).astype(np.uint8),
              weights=np.concatenate([x[1] for x in self.inputs]).astype(np.float32),
              origins=np.concatenate([x[2] for x in self.inputs]).astype(np.int8),
              gate=np.stack([x[3] for x in self.inputs]).astype(np.float32))
  np.savez(path,**arrays)
  write(Path(str(path)+'.json'),dict(status='PASS',events=len(self.inputs),
       semantics='Current G-NEAR routing only; independent cold-cache replay is nomination, not serving TPOT',
       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def main(a):
 rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
 assert os.environ['MGO_V2_PHYSICAL_GPUS']=='0,1,4,5'
 spec=json.loads(a.search_spec.read_text());assert spec['status']=='FROZEN'
 for c in spec['candidates']:
  assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
 warm=spec['warmup'];assert hashlib.sha256(Path(warm['path']).read_bytes()).hexdigest()==warm['sha256']
 cpus=affinity(rank)
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85)
 torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'));a.pcie_cpu_group=dist.new_group(backend='gloo')
 a.cell='R4_C30_B16_L512_O64';a.local_batch=16;a.capacities=[459,459,459,458];a.seed=42
 a.phase='COUNTERS';a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.debug_plan=True;a.live_routes=True
 a.pcie_native_controller=True;a.pcie_phase_diagnostic=False;a.pcie_capture_decisions=False;a.pcie_g2g_first_serial=True
 a.prefetch_off=True;a.prefill_optimized=True;a.prefill_layout_fast=True;a.decode_layout_fast=True;a.native_prefill=True
 a.expert_executor='native';a.capture_eviction_trace=False;a.record_main_eviction_trace=False
 a.policy='LA_CA_NEAR';a.pcie_quota_mode='rank_order';a.pcie_peer_costs=None
 a.prefill_diagnostic=False;a.post_generation_diagnostic=False;a.post_prefill_diagnostic=False
 cohort=a.output;model,backing,experts=load_model();rt=create_selected_runtime(a,model,backing,experts)
 grouped=GroupedDecode(rt);assert sum(a.capacities)+4*a.arena_budget==1843 and a.arena_budget==2
 assert rt.cache.numel()*rt.cache.element_size()==(a.capacities[rank]+2)*EB
 for row_count in (1,2):
  for slot in (0,1):
   with torch.inference_mode():
    w=rt.cache[slot];w.zero_();rt.kernel(torch.zeros((row_count,2048),device='cuda',dtype=torch.bfloat16),w[:1572864].view(768,2048),w[1572864:3145728].view(768,2048),w[3145728:].view(2048,768))
 torch.cuda.synchronize()
 def reset(arm,path,diagnostic=False):
  a.policy,a.pcie_quota_mode=ARMS[arm];a.output=path;a.pcie_phase_diagnostic=diagnostic
  path.mkdir(parents=True,exist_ok=True);rt.reset();rt.event_offset=0;rt.decode_layout_checks=0
  rt.gate_history=NativeGateHistory(48,128,128);rt.metadata=NativeLiveMetadata(16,rt.gate_history)
  grouped.reset(True);write(cohort/f'affinity_rank{rank}.json',configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt))
 def inputs(path):
  requests=json.loads(Path(path).read_text())['requests'];assert len(requests)==64
  assert len({r['source_row'] for r in requests})==64
  assert all(len(r['input_ids'])==512 for r in requests)
  rows=requests[rank*16:(rank+1)*16]
  return rows,torch.tensor([r['input_ids'] for r in rows],device='cuda')
 reset('R-NEAR',cohort/'prefill_validation');rows,ids=inputs(warm['path']);a.validate_prefill_optimized=True;begin(rt)
 checked=generate_live(model,rt,ids,1);state=validate_state(rt)
 assert rt.prefill_metadata_checks==48 and rt.prefill_layout_checks==48 and len(rt.prefill_numerics)==48 and rt.transport.calls==144
 write(cohort/f'prefill_validation_rank{rank}.json',dict(status='PASS',output=checked,state=state,per_layer_numerics=rt.prefill_numerics))
 a.validate_prefill_optimized=False;previous={}
 def generation(candidate,arm,repeat,n=64,diagnostic=False,nomination=False):
  path=cohort/candidate['candidate']/arm;reset(arm,path,diagnostic)
  a.phase='COUNTERS' if repeat==0 else 'MEASURE';a.validate_decode_layout=repeat==0
  grouped.validate=repeat==0;rt.metadata.validate_wire=repeat==0
  granular=GranularDiagnostic(rt,record_functions=False) if diagnostic else None
  assigned=ActualFetchCapture(rt) if diagnostic and rank==0 else None
  rows,ids=inputs(candidate['path']);begin(rt);empty=validate_state(rt)
  assert np.all(rt.keys<0) and not rt.controller.pending
  write(path/f'pinned_rank{rank}.json',rt.pinned_expert_store_receipt)
  if rank==0:write(cohort/'phase.json',dict(phase='warmup' if repeat==0 else 'target',candidate=candidate['candidate'],arm=arm,repeat=repeat,generation_started_unix=time.time()))
  capture=CurrentRoutingCapture(rt) if nomination and rank==0 else None
  gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  from torch._dynamo.utils import counters
  before=dict(counters['stats']);jit_before={f.__name__:sorted(f.cache[0]) for f in (projection,activate,weight_scatter,pack_current_metadata)}
  if repeat:
   with torch._dynamo.config.patch(error_on_recompile=True):result=generate_live(model,rt,ids,n)
  else:result=generate_live(model,rt,ids,n)
  if assigned:assigned.finish(path/'actual_fetches_repeat-1.npz')
  if granular:granular.close()
  if capture:capture.finish(path/'current_routing.npz')
  jit_after={f.__name__:sorted(f.cache[0]) for f in (projection,activate,weight_scatter,pack_current_metadata)}
  if repeat:assert jit_before==jit_after and before==dict(counters['stats']),'Compilation inside search generation'
  state=validate_state(rt);assert rt.transport.calls==n*96 and rt.index==n*48 and rt.debug_plan_checks==n*48
  assert state['physical_slots']==1843 and state['scheduler']['background_copies']==0
  trace=rt.policy.native_pcie.trace();assert trace.shape==(n*48,61)
  key=(candidate['candidate'],arm)
  if repeat and not nomination:
   signature=(result['argmax_hash'],state['state_hash'],hashlib.sha256(trace.tobytes()).hexdigest());expected=previous.get(key)
   assert expected is None or signature==expected,'Same-input search repetition changed tokens/cache/trace'
   previous[key]=signature
  quotas=np.column_stack((trace[:,19],trace[:,48:52])).astype(np.int64)
  assert np.all(quotas[:,0]==quotas[:,1:].sum(axis=1))
  if arm=='G-NEAR':assert np.all(np.abs(quotas[:,1:3].sum(axis=1)-quotas[:,3:5].sum(axis=1))<=1)
  np.save(path/f'policy_trace_repeat{repeat}_rank{rank}.npy',trace);np.save(path/f'quota_repeat{repeat}_rank{rank}.npy',quotas)
  if diagnostic:write(path/f'phases_repeat{repeat}_rank{rank}.json',rt.pcie_phase_trace)
  assert grouped.calls==(n-1)*48 and rt.metadata.calls==(n-1)*48
  if repeat==0:assert rt.decode_layout_checks==3024 and len(grouped.checks)==48 and rt.metadata.wire_checks==48
  else:assert rt.metadata.wire_checks==0
  write(path/f'grouped_repeat{repeat}_rank{rank}.json',grouped.receipt())
  result.update(candidate=candidate['candidate'],arm=arm,rank=rank,physical_gpu=[0,1,4,5][rank],repeat=repeat,
     phase='warmup' if repeat==0 else 'target',system='Ours',smoke=False,profiled=diagnostic or nomination,
     nomination=nomination,expert_executor='grouped_decode_native_prefill',metadata='native C++',policy=a.policy,quota_mode=a.pcie_quota_mode,
     expert_cache_start='empty',pcie_native_controller=True,pcie_g2g_first_serial=True,native_prefill=True,prefetch_off=True,
     validation=state,cache_before=empty,no_compile=before==dict(counters['stats']),triton_no_compile=jit_before==jit_after,
     debug_plan_checks=rt.debug_plan_checks,metadata_wire_checks=rt.metadata.wire_checks,
     peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
     pinned_host_bytes=rt.pinned_expert_store_receipt['bytes'],request_ids=[r['request_id'] for r in rows],
     forward_wire_bytes=rt.transport.forward_bytes,return_wire_bytes=rt.transport.return_bytes)
  write(path/f'repeat{repeat}_rank{rank}.json',result);dist.barrier(group=a.pcie_cpu_group)
  if rank==0:
   rr=[json.loads((path/f'repeat{repeat}_rank{r}.json').read_text()) for r in range(4)]
   assert len({r['release_ns'] for r in rr})==1
   release=rr[0]['release_ns'];first=max(r['first_ns'] for r in rr);end=max(r['end_ns'] for r in rr)
   row=dict(status='PASS',candidate=candidate['candidate'],arm=arm,repeat=repeat,TTFT=(first-release)/1e9,
      TPOT=(end-first)/1e9/(n-1),E2E=(end-release)/1e9,throughput=64*n/((end-release)/1e9),
      output_tokens=n,global_requests=64,smoke=False,profiled=diagnostic or nomination,nomination=nomination)
   write(path/f'repeat{repeat}.json',row);print(json.dumps(row),flush=True)
  dist.barrier(group=a.pcie_cpu_group)
 common=dict(candidate='_warmup',path=warm['path'])
 for arm in ('R-NEAR','G-NEAR'):generation(common,arm,0)
 candidates=spec['candidates'];repeats=1
 if a.search_stage=='nomination':
  for c in candidates:generation(c,'G-NEAR',1,n=16,nomination=True)
 elif a.search_stage=='screen':
  for j,c in enumerate(candidates):
   for arm in (('R-NEAR','G-NEAR') if j%2==0 else ('G-NEAR','R-NEAR')):generation(c,arm,1)
 else:
  assert len(candidates)==3
  for repeat in (1,2,3):
   for j,c in enumerate(candidates):
    for arm in (('R-NEAR','G-NEAR') if (j+repeat)%2 else ('G-NEAR','R-NEAR')):generation(c,arm,repeat)
  extra=torch.zeros(1,dtype=torch.int64)
  if rank==0:
   spreads={}
   for c in candidates:
    for arm in ('R-NEAR','G-NEAR'):
     values=[json.loads((cohort/c['candidate']/arm/f'repeat{r}.json').read_text())['TPOT'] for r in (1,2,3)]
     spreads[c['candidate']+'/'+arm]=(max(values)-min(values))/np.median(values)
   extra[0]=int(any(v>.05 for v in spreads.values()))
   write(cohort/'stability_decision.json',dict(threshold=.05,relative_tpot_spreads=spreads,additional_repeats=2 if extra[0] else 0,scope='ALL finalist arms'))
  dist.broadcast(extra,0,group=a.pcie_cpu_group);repeats=5 if extra.item() else 3
  if extra.item():
   for repeat in (4,5):
    for j,c in enumerate(candidates):
     for arm in (('R-NEAR','G-NEAR') if (j+repeat)%2 else ('G-NEAR','R-NEAR')):generation(c,arm,repeat)
  for c in candidates:
   for arm in ('G-NEAR','R-NEAR'):generation(c,arm,-1,diagnostic=True)
 if rank==0:
  for c in candidates:
   for arm in (('G-NEAR',) if a.search_stage=='nomination' else ('R-NEAR','G-NEAR')):
    path=cohort/c['candidate']/arm
    statistics={}
    if a.search_stage!='nomination':
     samples=[json.loads((path/f'repeat{r}.json').read_text()) for r in range(1,repeats+1)]
     for metric in ('TTFT','TPOT','E2E','throughput'):
      values=np.array([s[metric] for s in samples]);statistics[metric]=dict(median=float(np.median(values)),mean=float(values.mean()),
         sd=float(values.std(ddof=1)) if len(values)>1 else None,min=float(values.min()),max=float(values.max()),values=values.tolist())
    write(path/'result.json',dict(status='PASS',arm=arm,search_stage=a.search_stage,candidate=c['candidate'],
          primary_repeats=0 if a.search_stage=='nomination' else repeats,statistics=statistics,
          profiled=a.search_stage=='nomination',selection_stage=a.search_stage!='final',
          cell=a.cell,source='numa_shared_full_pinned',controller='native C++'))
  write(cohort/'result.json',dict(status='PASS',search_stage=a.search_stage,
       candidates=[c['candidate'] for c in candidates],primary_repeats=0 if a.search_stage=='nomination' else repeats,
       output_tokens=16 if a.search_stage=='nomination' else 64,selection_stage=a.search_stage!='final',
       search_spec=str(a.search_spec),search_spec_sha256=hashlib.sha256(a.search_spec.read_bytes()).hexdigest()))
 dist.barrier(group=a.pcie_cpu_group);grouped.close();rt.close();dist.destroy_process_group()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--search-spec',type=Path,required=True)
 p.add_argument('--search-stage',choices=('nomination','screen','final'),required=True);p.add_argument('--numa-shared-source-root',required=True)
 main(p.parse_args())
