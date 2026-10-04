"""Short full-model baseline gate for BR/CA/LA; never reports speed claims."""
from la_physical_worker import *
class PrefixRuntime(LiveRuntime):
 def execute(self,*args):
  recorder=getattr(self,"recorder",None)
  if recorder is None:return super().execute(*args)
  recorder.event=self.index
  with recorder.span("moe.layer"):return super().execute(*args)
 def reset(self):
  self.policy_kind={'BR':0,'CA':1,'LA':4}[self.args.policy]
  super().reset()

class ArenaBoundaryRuntime(PrefixRuntime):
 def __init__(self,a,model,backing,experts):
  super().__init__(a,model,backing,experts)
  self.cache=torch.empty((self.cap+a.arena_budget,EB//2),dtype=torch.bfloat16,device='cuda')
  self.keys=np.full(self.cap+a.arena_budget,-1,np.int32)
  self.h2d=PinnedH2DCache(self.cache,2)
 def reset(self):
  super().reset()
  from mgo_v2.cache import SlotArena
  self.arena=SlotArena(self.policy,self.args.arena_budget)
  self.prefill_boundary=None
 def execute(self,*args):
  result=super().execute(*args)
  if self.index==48:
   self.arena.assert_consistent();assert not self.arena.reservations
   assert np.all(self.keys[self.cap:]<0)
   self.prefill_boundary=dict(main_roles=self.cap,prefetch_roles=self.args.arena_budget,prefetch_empty=True,main_resident=int(np.count_nonzero(self.keys[:self.cap]>=0)))
  return result

class CompactBoundaryRuntime(ArenaBoundaryRuntime):
 def __init__(self,a,model,backing,experts):
  super().__init__(a,model,backing,experts)
  from mgo_v2.compact_metadata import CompactMetadata
  self.metadata=CompactMetadata(len(self.arrays['decode_origins'])//8)
  self.metadata_records=[]
 def reset(self):
  super().reset()
  self.metadata_records=[]
  if hasattr(self,'metadata'):self.metadata.calls=0
 def collect(self,layer,selected,weights,probs):
  from mgo_v2.runtime import gather_global_routes as legacy
  if self.index<48:return legacy(layer,selected,weights,probs)
  start=time.perf_counter();g=self.metadata.collect(self.index,selected,probs,self.arrays['gates'][self.index]);elapsed=time.perf_counter()-start
  start=time.perf_counter();reference=legacy(layer,selected,weights,probs);old_elapsed=time.perf_counter()-start
  assert np.array_equal(g.routes.selected_experts,reference.routes.selected_experts)
  assert np.array_equal(g.routes.origin_ranks,reference.routes.origin_ranks)
  assert g.counts==reference.counts
  assert np.array_equal(g.gate_scores,self.arrays['gates'][self.index])
  self.metadata_records.append(dict(compact_ms=elapsed*1000,legacy_ms=old_elapsed*1000))
  return g

def main(a):
 rank=int(os.environ['RANK']);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(rank)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=[3686//8+(r<3686%8) for r in range(8)];a.comm_mode='current';a.phase='COUNTERS'
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records)
 model,backing,experts=load_model();length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
 teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda');rows=[]
 for name in (['BR'] if a.instrument else ['BR','CA','LA']):
  a.policy=name
  from mgo_v2.decode_runtime import DecodeOffloadRuntime
  rt=(DecodeOffloadRuntime if a.split_controller else CompactBoundaryRuntime if a.compact else ArenaBoundaryRuntime if a.arena_budget else PrefixRuntime)(a,model,backing,experts);expected=None
  if a.compact:
   import la_physical_worker as live
   original_collect=live.gather_global_routes;live.gather_global_routes=rt.collect
  for repeat in range(2):
   if a.streaming:a.ready_first=bool(repeat)
   rt.reset()
   if a.instrument and repeat==1:
    from mgo_v2.phase_timing import PhaseRecorder
    import env_offload_worker as base
    import la_physical_worker as live
    recorder=PhaseRecorder();rt.recorder=recorder;rt.h2d.trace=recorder
    old_base,old_live=base.nvtx_phase,live.nvtx_phase;base.nvtx_phase=recorder.span;live.nvtx_phase=recorder.span
   row,tokens=generate(model,rt,ids,mask,teacher,8)
   if a.instrument and repeat==1:
    base.nvtx_phase,live.nvtx_phase=old_base,old_live
    write(a.output/f'phases_rank{rank}.json',recorder.receipt())
    rt.recorder=None;rt.h2d.trace=None
   if a.physical_prefetch:
    rt.h2d.synchronize()
    proof=json.loads((a.inputs/f'{name}_P{a.arena_budget}_proof.json').read_text())
    assert array_hash(rt.policy.slots[rank,:rt.cap])==proof['rank_state_hashes'][rank]
    assert rt.controller.counters==proof['counters']
    assert rt.h2d.metrics['copies']<=proof['max_copy_counts'][rank]
    baseline=json.loads((Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004/M0_gpu_gate.json').read_text())
    old=next(x for x in baseline['ranks'][rank]['rows'] if x['policy']==name)
    assert row['argmax_hash']==old['argmax_hash'],('baseline argmax mismatch',name,rank)
   else:
    assert array_hash(rt.keys[:rt.cap])==rt.proof['rank_state_hashes'][rank]
    assert rt.actual_h2d_bytes==rt.proof['H2D_bytes'][rank]
   assert not rt.mismatch.item()
   if expected is not None:assert np.array_equal(expected,tokens)
   expected=tokens
   rows.append(dict(policy=name,repeat=repeat,argmax_hash=row['argmax_hash'],state_hash=row['state_hash'],H2D_bytes=rt.actual_h2d_bytes,prefill_boundary=getattr(rt,'prefill_boundary',None),controller_times=getattr(rt,'controller_times',[]),debug_plan_checks=getattr(rt,'debug_plan_checks',0),scheduler_metrics=(rt.h2d.metrics if a.physical_prefetch else None),controller_counters=(rt.controller.counters if a.physical_prefetch else None),ready_first=getattr(a,'ready_first',False),ready_metrics=getattr(rt,'ready_metrics',None),metadata_records=getattr(rt,'metadata_records',[]),metadata_calls=(rt.metadata.calls if a.compact else None),metadata_record_bytes=(rt.metadata.record_bytes if a.compact else None)))
  if a.compact:live.gather_global_routes=original_collect
  if hasattr(rt,'close'):rt.close()
  # Detach closures before reclaiming the arena for the next policy.
  for block in model.model.layers:block.mlp.forward=lambda *args: None
  del rt;gc.collect();torch.cuda.empty_cache()
 write(a.output/f'rank{rank}.json',dict(status='PASS',rank=rank,horizon=8,rows=rows,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()));dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--instrument',action='store_true');p.add_argument('--arena-budget',type=int,default=0);p.add_argument('--compact',action='store_true');p.add_argument('--split-controller',action='store_true');p.add_argument('--debug-plan',action='store_true');p.add_argument('--physical-prefetch',action='store_true');p.add_argument('--streaming',action='store_true');main(p.parse_args())
