"""Two BR prefetch arms per model load; one clean primary and diagnostic each."""
from headline_ours_worker import *
from mgo_v2.selected_runtime import _create_runtime
from mgo_v2.decode_runtime import DecodeOffloadRuntime
from refactor_thread_affinity import configure
from br_prefetch_diagnostics import PrefetchDiagnostics
from br_prefetch_records import save_phase

def main(a):
 rank=int(os.environ['RANK']);assert int(os.environ['WORLD_SIZE'])==4
 physical=[0,1,4,5];assert os.environ['MGO_V2_PHYSICAL_GPUS']=='0,1,4,5'
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 spec=next(x for x in json.loads(Path(os.environ['MGO_HEADLINE_WORKLOADS']).read_text())['cells'] if x['cell']==a.cell)
 assert spec['expert_slots_per_rank']==[922,922,921,921]
 reference_root=Path(os.environ['MGO_BR_DIAGNOSTIC_REFERENCE']) if os.environ.get('MGO_BR_DIAGNOSTIC_REFERENCE') else None
 a.local_batch=spec['local_batch'];a.seed=42;a.policy='BR';a.phase='COUNTERS';a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.debug_plan=False;a.live_routes=True
 a.prefill_optimized=True;a.prefill_layout_fast=True;a.decode_layout_fast=False;a.validate_decode_layout=False;a.validate_prefill_optimized=False
 model,backing,experts=load_model();pinned_experts=pool=receipt=None;results=[]
 order=('on','off') if a.local_batch==16 else ('off','on')
 for arm in order:
  out=a.output/arm;out.mkdir(exist_ok=True);a.capacities=[x-2 for x in spec['expert_slots_per_rank']]
  options=selected_options();options.update(arena_budget=2,streaming=True,ready_first=True,trigger='T2',runtime_arm='BR_P2_OVERLAP')
  if pinned_experts is None:
   rt=_create_runtime(a,model,backing,experts,options);pinned_experts=rt.experts;pool=rt.pinned_expert_pool;receipt=rt.pinned_expert_store_receipt
  else:
   for key,value in options.items():setattr(a,key,value)
   a.direct_pinned_source=True;rt=DecodeOffloadRuntime(a,model,backing,pinned_experts);rt.pinned_expert_pool=pool;rt.pinned_expert_store_receipt=receipt
  if arm=='off':rt.prefetch_next=lambda:None
  assert rt.cache.numel()*rt.cache.element_size()==spec['expert_slots_per_rank'][rank]*EB
  assert a.streaming and a.ready_first and not getattr(a,'strict_serialized_phases',False) and not getattr(a,'post_expert_barrier',False)
  write(out/f'source_rank{rank}.json',dict(pinned=receipt,options=options,prefetch_enabled=arm=='on',capacities=a.capacities,physical_slots=spec['expert_slots_per_rank'],gpu=physical[rank]))
  for row_count in (1,2):
   for slot in (0,1):
    with torch.inference_mode():
     w=rt.cache[slot];w.zero_();rt.kernel(torch.zeros((row_count,2048),device='cuda',dtype=torch.bfloat16),w[:1572864].view(768,2048),w[1572864:3145728].view(768,2048),w[3145728:].view(2048,768))
  torch.cuda.synchronize()
  expected=None
  if reference_root is not None:
   expected=json.loads((reference_root/arm/f'primary_rank{rank}.json').read_text())
   assert expected['phase']=='primary' and expected['prefetch']==arm and expected['output_tokens']==257
   write(out/f'primary_rank{rank}.json',expected)
   if rank==0:
    original_primary=json.loads((reference_root/arm/'primary.json').read_text());assert original_primary['batch']==a.local_batch
    write(out/'primary.json',original_primary);results.append(original_primary)
  phases=(('warmup',9),('diagnostic',257)) if reference_root is not None else (('warmup',9),('primary',257),('diagnostic',257))
  for phase,n in phases:
   rt.reset();rt.event_offset=0;rt.gate_history=GateHistory(48,128,128)
   from mgo_v2.live_metadata import LiveMetadata
   rt.metadata=LiveMetadata(a.local_batch,rt.gate_history);configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
   rows=json.loads(Path(spec['warmup' if phase=='warmup' else 'target']['path']).read_text())['requests'];local=rows[rank*a.local_batch:(rank+1)*a.local_batch]
   ids=torch.tensor([r['input_ids'] for r in local],device='cuda');assert ids.shape==(a.local_batch,256)
   a.phase='COUNTERS' if phase=='warmup' else 'MEASURE';begin(rt);cache_before=validate_state(rt);assert np.all(rt.keys<0)
   # Lightweight byte snapshots only; no per-phase timers in the primary.
   original_execute=rt.execute;boundary={}
   def execute(*args,**kwargs):
    value=original_execute(*args,**kwargs)
    if rt.index==48:boundary.update(forward=rt.transport.forward_bytes,returned=rt.transport.return_bytes,h2d=rt.h2d.metrics['bytes'])
    return value
   rt.execute=execute
   if rank==0:write(a.output/'phase.json',dict(batch=a.local_batch,prefetch=arm,phase=phase,output_tokens=n))
   gc.collect();torch.cuda.synchronize()
   from torch._dynamo.utils import counters
   before=dict(counters['stats'])
   if phase=='diagnostic':rt.phase_diagnostic=PrefetchDiagnostics(rt);rt.phase_diagnostic.install(model)
   if phase=='warmup':result=generate_live(model,rt,ids,n)
   else:
    with torch._dynamo.config.patch(error_on_recompile=True):result=generate_live(model,rt,ids,n)
   if phase=='diagnostic':
    diagnostic=rt.phase_diagnostic.finish((result['end_ns']-result['release_ns'])/1e9);rt.phase_diagnostic=None
   rt.execute=original_execute
   validation=validate_state(rt);no_compile=before==dict(counters['stats']);assert phase=='warmup' or no_compile
   assert rt.transport.calls==n*48*2
   if arm=='off':assert validation['controller']['issued']==0 and validation['controller']['promotions']==0
   result.update(phase=phase,prefetch=arm,rank=rank,gpu=physical[rank],validation=validation,no_compile=no_compile,prefill_bytes=boundary,decode_bytes=dict(forward=rt.transport.forward_bytes-boundary['forward'],returned=rt.transport.return_bytes-boundary['returned'],h2d=rt.h2d.metrics['bytes']-boundary['h2d']),ready_metrics=rt.ready_metrics)
   if phase=='primary':expected=result
   if phase=='diagnostic':
    assert result['tokens']==expected['tokens'] and validation['state_hash']==expected['validation']['state_hash'] and validation['role_hash']==expected['validation']['role_hash']
    assert result['decode_bytes']==expected['decode_bytes']
    diagnostic.update(token_cache_byte_parity=True,output=result)
   receipt_phase=save_phase(out,phase,rank,result,diagnostic if phase=='diagnostic' else None);dist.barrier()
   if rank==0:
    rr=[json.loads((out/f'{receipt_phase}_rank{r}.json').read_text()) for r in range(4)];assert len({x['release_ns'] for x in rr})==1
    start=rr[0]['release_ns'];first=max(x['first_ns'] for x in rr);end=max(x['end_ns'] for x in rr)
    summary=dict(status='PASS',batch=a.local_batch,prefetch=arm,phase=phase,TTFT=(first-start)/1e9,TPOT=(end-first)/1e9/(n-1),E2E=(end-start)/1e9,decode_steps=n-1,decode_peer_bytes=sum(x['decode_bytes']['forward']+x['decode_bytes']['returned'] for x in rr),decode_h2d_bytes=sum(x['decode_bytes']['h2d'] for x in rr))
    write(out/f'{phase}.json',summary);print(json.dumps(summary),flush=True)
    if phase=='primary':results.append(summary)
   dist.barrier()
  rt.close()
  for block in model.model.layers:block.mlp.forward=lambda *unused:None
  del original_execute,execute,rt;gc.collect();torch.cuda.empty_cache();dist.barrier()
 if rank==0:write(a.output/'result.json',dict(status='PASS',batch=a.local_batch,primary_reference=str(reference_root) if reference_root else None,results=results,scope='one primary and separate full diagnostic per arm; OFF retains identical overlap and MAIN capacity; model and pinned source reused'))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
