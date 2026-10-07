"""H0 versus native TTFT/TPOT with frozen prefill/decode routing and tokens."""
from headline_ours_worker import *
from mgo_v2.selected_runtime import _create_runtime
from mgo_v2.native_expert import NativeExpertExecutor
from refactor_thread_affinity import configure
from mgo_v2.live_metadata import LiveMetadata


def generate_fixed(model,rt,ids,teacher,n):
 dist.barrier();torch.cuda.synchronize()
 release=torch.tensor([time.perf_counter_ns()+200_000_000 if dist.get_rank()==0 else 0],device='cuda',dtype=torch.int64)
 dist.broadcast(release,0);start=int(release.item())
 while time.perf_counter_ns()<start:time.sleep(.0001)
 mask=torch.ones_like(ids);past=None;tokens=[];finite=torch.ones((),dtype=torch.bool,device='cuda');stamps=[]
 with torch.inference_mode():
  for step in range(n):
   pos=torch.arange(mask.shape[1]-ids.shape[1],mask.shape[1],device='cuda')[None,:].expand(len(ids),-1)
   out=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=past,use_cache=True,logits_to_keep=1)
   past=out.past_key_values;tokens.append(out.logits[:,-1].argmax(-1));finite.logical_and_(torch.isfinite(out.logits).all())
   torch.cuda.synchronize();stamps.append(time.perf_counter_ns())
   if step<n-1:ids=teacher[:,step:step+1];mask=torch.cat((mask,mask.new_ones((len(ids),1))),1)
 actual=torch.stack(tokens,dim=1).cpu().numpy();assert bool(finite) and rt.index==48*n
 return dict(release_ns=start,first_ns=stamps[0],end_ns=stamps[-1],token_ready_ns=stamps,finite_logits=True,output_tokens=n,argmax_hash=array_hash(actual),tokens=actual.tolist())


def main(a):
 rank=int(os.environ['RANK']);physical=[0,1,4,5]
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42)
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 spec=next(x for x in json.loads(Path(os.environ['MGO_HEADLINE_WORKLOADS']).read_text())['cells'] if x['cell']==a.cell)
 a.local_batch=spec['local_batch'];a.seed=42;a.policy='BR';a.phase='COUNTERS';a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.debug_plan=False;a.live_routes=True
 a.prefill_optimized=True;a.prefill_layout_fast=True;a.decode_layout_fast=True;a.validate_decode_layout=False;a.validate_prefill_optimized=False
 a.capacities=spec.get('expert_slots_per_rank',[461,461,461,460])
 options=selected_options();options.update(arena_budget=2,streaming=True,ready_first=True,trigger='T2',runtime_arm='BR_P2_OVERLAP')
 native=NativeExpertExecutor()
 model,backing,experts=load_model();rt=_create_runtime(a,model,backing,experts,options);rt.prefetch_next=lambda:None
 write(a.output/f'source_rank{rank}.json',dict(options=options,capacities=a.capacities,gpu=physical[rank],prefetch=False,pinned=rt.pinned_expert_store_receipt))
 def reset():
  rt.reset();rt.event_offset=0;rt.gate_history=GateHistory(48,128,128);rt.metadata=LiveMetadata(a.local_batch,rt.gate_history)
  configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt);begin(rt);assert np.all(rt.keys<0)
 def inputs(kind):
  rows=json.loads(Path(spec[kind]['path']).read_text())['requests'][rank*a.local_batch:(rank+1)*a.local_batch]
  return torch.tensor([r['input_ids'] for r in rows],device='cuda')
 warm=inputs('warmup');target=inputs('target')
 for name,executor in [('h0',None),('native',native)]:
  rt.native_executor=executor;reset()
  if rank==0:write(a.output/'phase.json',dict(phase='warmup',backend=name))
  generate_live(model,rt,warm,min(4,a.decode_steps+1));validate_state(rt)
 rt.native_executor=None;reset();routes=[];original=rt.execute
 def capture(layer,hidden,selected,weights,probs):
  routes.append((selected.detach().clone(),weights.detach().clone(),probs.detach().clone()))
  return original(layer,hidden,selected,weights,probs)
 rt.execute=capture
 if rank==0:write(a.output/'phase.json',dict(phase='capture',decode_steps=a.decode_steps))
 captured=generate_live(model,rt,target,a.decode_steps+1);validate_state(rt);rt.execute=original
 assert len(routes)==(a.decode_steps+1)*48
 teacher=torch.tensor(captured['tokens'],device='cuda');route_bytes=sum(t.numel()*t.element_size() for row in routes for t in row)
 # Store a content hash, not a future-dependent controller. Only current
 # event tensors are substituted; predictor is disabled in both arms.
 h=hashlib.sha256()
 for row in routes:
  for t in row:h.update(t.cpu().view(torch.uint8).numpy().tobytes())
 write(a.output/f'trace_rank{rank}.json',dict(route_sha256=h.hexdigest(),events=len(routes),bytes=route_bytes,teacher_tokens=captured['tokens']))
 proof=None;baseline_tokens=None;results=[]
 for run,name in enumerate(('h0','native','native','h0')):
  rt.native_executor=native if name=='native' else None;reset();a.phase='MEASURE'
  boundary={};before_waves=native.waves;before_groups=native.groups
  def frozen(layer,hidden,selected,weights,probs):
   sel,wei,pro=routes[rt.index];value=original(layer,hidden,sel,wei,pro)
   if rt.index==48:boundary.update(forward=rt.transport.forward_bytes,returned=rt.transport.return_bytes,h2d=rt.h2d.metrics['bytes'])
   return value
  rt.execute=frozen;gc.collect();torch.cuda.synchronize()
  if rank==0:write(a.output/'phase.json',dict(phase='primary',backend=name,run=run,decode_steps=a.decode_steps))
  from torch._dynamo.utils import counters
  before=dict(counters['stats'])
  with torch._dynamo.config.patch(error_on_recompile=True):result=generate_fixed(model,rt,target,teacher,a.decode_steps+1)
  rt.execute=original;validation=validate_state(rt);assert before==dict(counters['stats'])
  byte_counts=dict(forward=rt.transport.forward_bytes-boundary['forward'],returned=rt.transport.return_bytes-boundary['returned'],h2d=rt.h2d.metrics['bytes']-boundary['h2d'])
  current=dict(state=validation['state_hash'],roles=validation['role_hash'],controller=validation['controller'],prefill_h2d_bytes=boundary['h2d'],decode_bytes=byte_counts)
  if proof is None:proof=current;baseline_tokens=np.asarray(result['tokens'])
  assert current==proof,'frozen logical/byte parity failed'
  if name=='h0':assert np.array_equal(np.asarray(result['tokens']),baseline_tokens)
  result.update(status='PASS',rank=rank,backend=name,run=run,validation=validation,decode_bytes=byte_counts,no_compile=True,token_agreement_to_h0=float(np.mean(np.asarray(result['tokens'])==baseline_tokens)),native_waves=native.waves-before_waves,native_groups=native.groups-before_groups,ready_metrics=dict(rt.ready_metrics),frozen_parity=True)
  write(a.output/f'run{run}_rank{rank}.json',result);dist.barrier()
  if rank==0:
   rr=[json.loads((a.output/f'run{run}_rank{r}.json').read_text()) for r in range(4)];start=rr[0]['release_ns'];assert len({r['release_ns'] for r in rr})==1
   first=max(r['first_ns'] for r in rr);end=max(r['end_ns'] for r in rr)
   row=dict(status='PASS',backend=name,run=run,TTFT=(first-start)/1e9,TPOT=(end-first)/1e9/a.decode_steps,E2E=(end-start)/1e9,token_agreement_to_h0=float(np.mean([r['token_agreement_to_h0'] for r in rr])),decode_peer_bytes=sum(r['decode_bytes']['forward']+r['decode_bytes']['returned'] for r in rr),decode_h2d_bytes=sum(r['decode_bytes']['h2d'] for r in rr),scope=f'{a.decode_steps} decode forwards; frozen H0 prefill/decode routes and teacher inputs')
   results.append(row);write(a.output/f'run{run}.json',row);print(json.dumps(row),flush=True)
  dist.barrier()
 rt.close()
 if rank==0:write(a.output/'result.json',dict(status='PASS',results=results,route_frozen=True,production_default_changed=False,cell=a.cell,decode_steps=a.decode_steps))
 dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--decode-steps',type=int,default=16);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
