"""H0 versus native TTFT/TPOT with frozen prefill/decode routing and tokens."""
from headline_ours_worker import *
from mgo_v2.selected_runtime import _create_runtime
from mgo_v2.native_expert import NativeExpertExecutor
from refactor_thread_affinity import configure
from mgo_v2.live_metadata import LiveMetadata
from native_trace_counts import distinct_event_counts


def generate_fixed(model,rt,ids,teacher,n):
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
   past=out.past_key_values;tokens.append(out.logits[:,-1].argmax(-1));finite.logical_and_(torch.isfinite(out.logits).all())
   torch.cuda.synchronize();stamps.append(time.perf_counter_ns())
   if step<n-1:ids=teacher[:,step:step+1];mask=torch.cat((mask,mask.new_ones((len(ids),1))),1)
 if getattr(rt,'phase_diagnostic',None):rt.phase_diagnostic.stop()
 actual=torch.stack(tokens,dim=1).cpu().numpy();assert bool(finite) and rt.index==48*n
 return dict(release_ns=start,first_ns=stamps[0],end_ns=stamps[-1],token_ready_ns=stamps,finite_logits=True,output_tokens=n,argmax_hash=array_hash(actual),tokens=actual.tolist())


def main(a):
 rank=int(os.environ['RANK']);physical=[0,1,4,5]
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 # The rank bootstrap supplies an explicit "0". For the normal NVSwitch
 # control, restore the user's requested unset state before NCCL initializes.
 if os.environ.get('NCCL_P2P_DISABLE')=='0':os.environ.pop('NCCL_P2P_DISABLE')
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42)
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 spec=next(x for x in json.loads(Path(os.environ['MGO_HEADLINE_WORKLOADS']).read_text())['cells'] if x['cell']==a.cell)
 a.local_batch=spec['local_batch'];a.seed=42;a.policy=os.environ.get('MGO_NATIVE_POLICY','BR');a.phase='COUNTERS';a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.debug_plan=False;a.live_routes=True
 assert a.policy in ('BR','CA','CA_NATIVE','LA_CA_NEAR')
 prefetch=os.environ.get('MGO_NATIVE_PREFETCH','off');assert prefetch in ('on','off')
 compare_prefetch=os.environ.get('MGO_NATIVE_COMPARE_PREFETCH','0')=='1'
 policy_comparison=os.environ.get('MGO_NATIVE_COMPARE_POLICY','0')
 assert policy_comparison in ('0','1','CA_NEAR','BR_CA','CA_NATIVE','TRIPLE')
 compare_policy=policy_comparison!='0'
 policy_modes={'1':('BR','LA_CA_NEAR'),'CA_NEAR':('LA_CA_NEAR','CA'),'BR_CA':('BR','CA'),'CA_NATIVE':('CA','CA_NATIVE'),'TRIPLE':('BR','CA_NATIVE','LA_CA_NEAR')}.get(policy_comparison,())
 capture_policy='LA_CA_NEAR' if policy_comparison in ('BR_CA','CA_NATIVE','TRIPLE') else (policy_modes[0] if compare_policy else a.policy)
 sync_ablation=os.environ.get('MGO_NATIVE_SYNC_ABLATION','0')=='1'
 assert not (compare_prefetch and compare_policy)
 if compare_prefetch:assert a.policy=='LA_CA_NEAR' and prefetch=='on'
 if compare_policy:assert a.policy==capture_policy and prefetch=='off'
 if sync_ablation:assert policy_comparison=='BR_CA' and prefetch=='off'
 comparing=compare_prefetch or compare_policy
 a.prefill_optimized=True;a.prefill_layout_fast=True;a.decode_layout_fast=True;a.native_prefill=True;a.validate_decode_layout=False;a.validate_prefill_optimized=False
 a.prefetch_off=prefetch=='off';a.collective_barrier_ablation=sync_ablation
 a.capacities=spec.get('expert_slots_per_rank',[461,461,461,460])
 if os.environ.get('MGO_NATIVE_MAIN_CAPACITY','0')=='1':
  a.capacities=[x-2 for x in a.capacities]
 options=selected_options();options.update(arena_budget=2,streaming=True,ready_first=True,trigger='T2',runtime_arm='BR_P2_OVERLAP')
 native=NativeExpertExecutor()
 model,backing,experts=load_model();rt=_create_runtime(a,model,backing,experts,options)
 prefetch_on=rt.prefetch_next
 if prefetch=='off':rt.prefetch_next=lambda:None
 write(a.output/f'source_rank{rank}.json',dict(options=options,capacities=a.capacities,gpu=physical[rank],prefetch=prefetch,policy=a.policy,capture_policy=capture_policy,collective_barrier_ablation=sync_ablation,nccl_p2p_disable=os.environ.get('NCCL_P2P_DISABLE'),nccl_ib_disable=os.environ.get('NCCL_IB_DISABLE'),pinned=rt.pinned_expert_store_receipt))
 def reset():
  rt.reset();rt.event_offset=0;rt.gate_history=GateHistory(48,128,128);rt.metadata=LiveMetadata(a.local_batch,rt.gate_history)
  configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt);begin(rt);assert np.all(rt.keys<0)
 def inputs(kind):
  rows=json.loads(Path(spec[kind]['path']).read_text())['requests'][rank*a.local_batch:(rank+1)*a.local_batch]
  return torch.tensor([r['input_ids'] for r in rows],device='cuda')
 warm=inputs('warmup');target=inputs('target')
 warm_modes=[('prefetch_on',native),('prefetch_off',native)] if compare_prefetch else ([(name,native) for name in policy_modes] if compare_policy else [('h0',None),('native',native)])
 for name,executor in warm_modes:
  if compare_policy:a.policy=name
  rt.native_executor=executor;reset()
  if compare_prefetch:rt.prefetch_next=prefetch_on if name=='prefetch_on' else lambda:None
  if rank==0:write(a.output/'phase.json',dict(phase='warmup',backend=name))
  generate_live(model,rt,warm,min(4,a.decode_steps+1));validate_state(rt)
 if compare_policy:a.policy=capture_policy
 rt.native_executor=native if compare_policy else None;reset();routes=[];original=rt.execute
 if compare_prefetch:rt.prefetch_next=prefetch_on
 def capture(layer,hidden,selected,weights,probs):
  routes.append((selected.detach().clone(),weights.detach().clone(),probs.detach().clone()))
  return original(layer,hidden,selected,weights,probs)
 rt.execute=capture
 if rank==0:write(a.output/'phase.json',dict(phase='capture',decode_steps=a.decode_steps))
 captured=generate_live(model,rt,target,a.decode_steps+1);validate_state(rt);rt.execute=original
 assert len(routes)==(a.decode_steps+1)*48
 active_total=active_decode=None
 if comparing:
  sizes=[row[0].numel() for row in routes]
  local_routes=torch.cat([row[0].reshape(-1).to(torch.uint8) for row in routes])
  gathered=[torch.empty_like(local_routes) for _ in range(4)]
  dist.all_gather(gathered,local_routes)
  rank_routes=[row.cpu().numpy() for row in gathered]
  distinct=distinct_event_counts(rank_routes,sizes)
  active_total=int(distinct.sum());active_decode=int(distinct[48:].sum())
 teacher=torch.tensor(captured['tokens'],device='cuda');route_bytes=sum(t.numel()*t.element_size() for row in routes for t in row)
 # Store a content hash, not a future-dependent controller. Only current
 # event tensors are substituted; ON/OFF keeps predictor access causal.
 h=hashlib.sha256()
 for row in routes:
  for t in row:h.update(t.cpu().view(torch.uint8).numpy().tobytes())
 write(a.output/f'trace_rank{rank}.json',dict(route_sha256=h.hexdigest(),events=len(routes),bytes=route_bytes,teacher_tokens=captured['tokens']))
 proofs={};baseline_tokens=None;results=[]
 modes=('prefetch_on','prefetch_off','prefetch_off','prefetch_on') if compare_prefetch else (policy_modes if policy_comparison=='TRIPLE' else ((policy_modes[0],policy_modes[1],policy_modes[1],policy_modes[0]) if compare_policy else ('h0','native','native','h0')))
 for run,name in enumerate(modes):
  if compare_policy:a.policy=name
  rt.native_executor=native if comparing or name=='native' else None;reset();a.phase='MEASURE'
  if compare_prefetch:rt.prefetch_next=prefetch_on if name=='prefetch_on' else lambda:None
  boundary={};before_waves=native.waves;before_groups=native.groups
  def frozen(layer,hidden,selected,weights,probs):
   sel,wei,pro=routes[rt.index];value=original(layer,hidden,sel,wei,pro)
   if rt.index==48:boundary.update(forward=rt.transport.forward_bytes,returned=rt.transport.return_bytes,h2d=rt.h2d.metrics['bytes'],mandatory=rt.controller.counters['mandatory'])
   return value
  rt.execute=frozen;gc.collect();torch.cuda.synchronize()
  if rank==0:write(a.output/'phase.json',dict(phase='primary',backend=name,run=run,decode_steps=a.decode_steps))
  from torch._dynamo.utils import counters
  before=dict(counters['stats'])
  with torch._dynamo.config.patch(error_on_recompile=True):result=generate_fixed(model,rt,target,teacher,a.decode_steps+1)
  rt.execute=original;validation=validate_state(rt);assert before==dict(counters['stats'])
  expected_barriers=48*a.decode_steps if sync_ablation else 0
  assert rt.h2d_global_barriers==rt.post_expert_barriers==expected_barriers
  byte_counts=dict(forward=rt.transport.forward_bytes-boundary['forward'],returned=rt.transport.return_bytes-boundary['returned'],h2d=rt.h2d.metrics['bytes']-boundary['h2d'])
  current=dict(state=validation['state_hash'],roles=validation['role_hash'],controller=validation['controller'],prefill_h2d_bytes=boundary['h2d'],decode_bytes=byte_counts)
  if baseline_tokens is None:baseline_tokens=np.asarray(result['tokens'])
  if name in proofs:assert current==proofs[name],'same-mode logical/byte parity failed'
  else:proofs[name]=current
  if name=='h0':assert np.array_equal(np.asarray(result['tokens']),baseline_tokens)
  hit=dict(active_distinct=active_total,active_decode_distinct=active_decode,mandatory=validation['controller']['mandatory'],mandatory_decode=validation['controller']['mandatory']-boundary['mandatory'],main_hit_rate=1-validation['controller']['mandatory']/active_total,decode_main_hit_rate=1-(validation['controller']['mandatory']-boundary['mandatory'])/active_decode) if comparing else None
  result.update(status='PASS',rank=rank,backend=name,run=run,validation=validation,decode_bytes=byte_counts,hit=hit,no_compile=True,reference_backend=modes[0],token_agreement_to_reference=float(np.mean(np.asarray(result['tokens'])==baseline_tokens)),native_waves=native.waves-before_waves,native_groups=native.groups-before_groups,ready_metrics=dict(rt.ready_metrics),frozen_parity=True,dispatch_barriers=rt.h2d_global_barriers,return_barriers=rt.post_expert_barriers)
  write(a.output/f'run{run}_rank{rank}.json',result);dist.barrier()
  if rank==0:
   rr=[json.loads((a.output/f'run{run}_rank{r}.json').read_text()) for r in range(4)];start=rr[0]['release_ns'];assert len({r['release_ns'] for r in rr})==1
   first=max(r['first_ns'] for r in rr);end=max(r['end_ns'] for r in rr)
   row=dict(status='PASS',backend=name,run=run,TTFT=(first-start)/1e9,TPOT=(end-first)/1e9/a.decode_steps,E2E=(end-start)/1e9,reference_backend=modes[0],token_agreement_to_reference=float(np.mean([r['token_agreement_to_reference'] for r in rr])),decode_peer_bytes=sum(r['decode_bytes']['forward']+r['decode_bytes']['returned'] for r in rr),decode_h2d_bytes=sum(r['decode_bytes']['h2d'] for r in rr),main_hit_rate=rr[0]['hit']['main_hit_rate'] if comparing else None,decode_main_hit_rate=rr[0]['hit']['decode_main_hit_rate'] if comparing else None,mandatory=rr[0]['hit']['mandatory'] if comparing else None,prefetch_issued=rr[0]['validation']['controller']['issued'] if comparing else None,scope=f'{a.decode_steps} decode forwards; frozen routing and teacher inputs')
   results.append(row);write(a.output/f'run{run}.json',row);print(json.dumps(row),flush=True)
  dist.barrier()
 diagnostic_prefix=int(os.environ.get('MGO_NATIVE_POLICY_DIAGNOSTIC','0'))
 if diagnostic_prefix:
  assert compare_policy and 1<=diagnostic_prefix<=a.decode_steps
  from generation_phase_diagnostics import GenerationDiagnostics
  for name in policy_modes:
   a.policy=name;rt.native_executor=native;rt.prefetch_next=lambda:None;reset();a.phase='MEASURE'
   def frozen_diagnostic(layer,hidden,selected,weights,probs):
    sel,wei,pro=routes[rt.index]
    return original(layer,hidden,sel,wei,pro)
   rt.execute=frozen_diagnostic
   rt.phase_diagnostic=GenerationDiagnostics(rt);rt.phase_diagnostic.install(model)
   if rank==0:write(a.output/'phase.json',dict(phase='policy_diagnostic',backend=name,decode_steps=diagnostic_prefix))
   gc.collect();torch.cuda.synchronize()
   with torch._dynamo.config.patch(error_on_recompile=True):check=generate_fixed(model,rt,target,teacher,diagnostic_prefix+1)
   diagnostic=rt.phase_diagnostic.finish((check['end_ns']-check['release_ns'])/1e9)
   expected_barriers=48*diagnostic_prefix if sync_ablation else 0
   assert rt.h2d_global_barriers==rt.post_expert_barriers==expected_barriers
   rt.phase_diagnostic=None;rt.execute=original
   expected=next(json.loads((a.output/f'run{run}_rank{rank}.json').read_text()) for run,mode in enumerate(modes) if mode==name)
   assert check['tokens']==[row[:diagnostic_prefix+1] for row in expected['tokens']]
   diagnostic.update(policy=name,decode_steps=diagnostic_prefix,token_prefix_parity=True,validation=validate_state(rt),output=dict(release_ns=check['release_ns'],first_ns=check['first_ns'],end_ns=check['end_ns'],token_ready_ns=check['token_ready_ns']))
   write(a.output/f'diagnostic_{name}_rank{rank}.json',diagnostic)
   dist.barrier()
 rt.close()
 if rank==0:write(a.output/'result.json',dict(status='PASS',results=results,route_frozen=True,production_default_changed=False,cell=a.cell,decode_steps=a.decode_steps,prefetch=prefetch,capture_policy=capture_policy,collective_barrier_ablation=sync_ablation,policy_compare=compare_policy,policy_modes=list(policy_modes),active_total=active_total,active_decode=active_decode,main_capacities=a.capacities,nccl_p2p_disable=os.environ.get('NCCL_P2P_DISABLE')))
 dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--decode-steps',type=int,default=int(os.environ.get('MGO_NATIVE_DECODE_STEPS','16')));p.add_argument('--output',type=Path,required=True);main(p.parse_args())
