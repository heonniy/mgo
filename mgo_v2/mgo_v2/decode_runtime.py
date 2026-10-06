"""Shared physical decode runtime; later milestones enable prefetch and overlap.

This module owns integration, controller.py owns logical state, and cache.py owns
C+P role mappings. The old runtime remains the diagnostic parity reference.
"""
import hashlib,pickle,time
import numpy as np
import torch
import torch.distributed as dist
from la_physical_worker import LiveRuntime,EB,device_layout,nvtx_phase,plan_layout,PinnedH2DCache
from .runtime import gather_global_routes
from .controller import DecodePrefetchController
from .predictor import TransitionPredictor
from .compact_metadata import CompactMetadata
from .pinned_h2d import PriorityH2DScheduler
from .ready_compute import expert_order
from .fused_transport import FusedTokenRankTransport
from .coalesced_return import combine_rank_partials

class DecodeOffloadRuntime(LiveRuntime):
 def __init__(self,a,model,backing,experts):
  self.predictor=TransitionPredictor(np.load('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/predictor/transition.npy'))
  super().__init__(a,model,backing,experts)
  self.cache=torch.empty((self.cap+a.arena_budget,EB//2),dtype=torch.bfloat16,device='cuda');self.keys=np.full(self.cap+a.arena_budget,-1,np.int32)
  self.h2d=PinnedH2DCache(self.cache,2)
  if getattr(a,'physical_prefetch',False):self.h2d=PriorityH2DScheduler(self.cache,staging_backend=getattr(self.args,"staging_backend","torch"),cpu_team=getattr(self.args,"staging_cpu_team",None))
  self.metadata=CompactMetadata(len(self.arrays['decode_origins'])//self.world,async_inputs=getattr(a,'async_metadata_inputs',False))
 def stage_frozen_inputs(self,horizon):
  # Benchmark input delivery is common across strategies and outside timing.
  # Only the current event is exposed to the runtime; prediction never reads
  # future routes even though the frozen input storage is device-resident.
  ids=[];weights=[];sizes=[]
  for event in range((horizon+1)*48):
   lo=int(self.arrays['offsets'][event]);start,end=self.ranges[0 if event<48 else 1]
   ids.append(self.arrays['selected'][lo+start:lo+end]);weights.append(self.arrays['weights'][lo+start:lo+end]);sizes.append(end-start)
  all_ids=torch.tensor(np.concatenate(ids),device='cuda',dtype=torch.int64)
  all_weights=torch.tensor(np.concatenate(weights),device='cuda',dtype=torch.bfloat16)
  self.route_tensors=list(zip(all_ids.split(sizes),all_weights.split(sizes)))
  self.frozen_input_bytes=all_ids.numel()*all_ids.element_size()+all_weights.numel()*all_weights.element_size()
 def reset(self):
  if isinstance(getattr(self,'h2d',None),PriorityH2DScheduler):
   self.h2d.close();self.h2d=PriorityH2DScheduler(self.cache,staging_backend=getattr(self.args,"staging_backend","torch"),cpu_team=getattr(self.args,"staging_cpu_team",None))
  super().reset()
  self.controller=DecodePrefetchController(self.args.capacities,self.args.arena_budget,self.args.policy,self.args.seed,self.predictor)
  self.policy=self.controller.main;self.arena=self.controller.arena
  self.prefill_boundary=None;self.controller_times=[];self.debug_plan_checks=0;self.ready_metrics=dict(waits=0,ready_before_first_wait=0);self.unique_combine_layers=0
  self.transport=FusedTokenRankTransport('exact')
  if hasattr(self,'metadata'):self.metadata.calls=0
 def plan_event(self,layer,selected,weights,probs):
  with nvtx_phase('moe.metadata'):
   if self.index<48:g=gather_global_routes(layer,selected,weights,probs)
   else:g=self.metadata.collect(self.index,selected,probs,self.arrays['gates'][self.index])
  r=g.routes;self.current_histogram=getattr(g,'histogram',None)
  with nvtx_phase('moe.current_controller'):
   start=time.perf_counter()
   out,promotions,discards=self.controller.plan_current(self.index,r.selected_experts,r.routing_weights,r.origin_ranks,self.arrays['gates'][self.index])
   if self.index>=48 and self.args.phase!='MEASURE':self.controller_times.append((time.perf_counter()-start)*1000)
   targets,effective,masses,lengths,destinations,fetches,row=out
   if getattr(self.args,'physical_prefetch',False):
    for promotion in promotions:
     if promotion.rank==self.rank:
      assert self.keys[promotion.promoted_physical_slot]==promotion.key
      assert self.keys[promotion.recycled_physical_slot]==promotion.victim
      self.h2d.promote(promotion.promoted_physical_slot,promotion.key)
      self.h2d.invalidate(promotion.recycled_physical_slot,promotion.victim)
      self.keys[promotion.recycled_physical_slot]=-1
    for rank,slot,key in discards:
     if rank==self.rank:self.h2d.discard(slot,key);self.keys[slot]=-1
   else:
    assert not promotions and not discards
    assert [list(f) for f in fetches]==self.reference[self.index]
   self.current_global_fetch_count=len(fetches)
   e=plan_layout(effective,lengths,destinations,r.origin_ranks,g.counts,self.rank)
   e.update(layer=layer,targets=targets,selected=selected.cpu().numpy(),fetches=[(key,int(self.arena.main_physical[rank][slot]),victim,rep) for rank,key,slot,victim,rep in fetches if rank==self.rank])
   e['groups']=[(expert,rows,cols,int(self.arena.main_physical[self.rank][np.flatnonzero(self.policy.slots[self.rank]==layer*128+expert)[0]])) for expert,rows,cols in e['groups']]
  if getattr(self.args,'debug_plan',False):
   payload=(fetches,self.policy.owner,self.policy.slots,self.arena.main_physical,self.arena.prefetch_physical)
   digest=np.frombuffer(hashlib.sha256(pickle.dumps(payload,protocol=4)).digest(),np.uint8).copy()
   tensor=torch.tensor(digest,device='cuda');all_hashes=torch.empty((self.world,32),dtype=torch.uint8,device='cuda')
   dist.all_gather_into_tensor(all_hashes.view(-1),tensor);assert bool((all_hashes==all_hashes[0]).all())
   self.debug_plan_checks+=1
  if self.args.phase!='MEASURE' and self.index%768==0:print(f'{self.args.policy} event={self.index} split_controller',flush=True)
  if getattr(self.args,'unique_combine',False) and self.index>=48:
   from .unique_combine import validate_unique_layout
   e['unique_combine_validated']=validate_unique_layout(e,len(self.arrays['decode_origins'])//self.world)
   self.unique_combine_layers+=int(e['unique_combine_validated'])
  return device_layout(e)
 def prefetch_next(self):
  if self.index<48:return
  with nvtx_phase('moe.prefetch_controller'):
   reservations=self.controller.plan_prefetch_next(self.current_histogram)
  for rank,key,pfslot in reservations:
   if rank==self.rank:
    physical=int(self.arena.prefetch_physical[rank][pfslot])
    self.h2d.enqueue_prefetch(physical,key,self.experts[key]);self.keys[physical]=key
 def apply_fetches(self,e):
  if not getattr(self.args,'physical_prefetch',False):return super().apply_fetches(e)
  for key,slot,victim,rep in e['fetches']:
   assert not rep and self.keys[slot]==victim,(self.index,key,slot,victim,self.keys[slot])
   self.h2d.enqueue_demand(slot,key,self.experts[key]);self.keys[slot]=key
 def compute(self,packet,e,layer):
  if self.index<48 or not (getattr(self.args,'streaming',False) or getattr(self.args,'fused',False)):return super().compute(packet,e,layer)
  mode,received,_,rw=packet;assert mode=='current'
  groups=e['groups'];parts=[None]*len(groups)
  for i in expert_order(groups,self.h2d,getattr(self.args,'ready_first',False),self.ready_metrics):
   expert,rows,cols,slot=groups[i];assert self.keys[slot]==layer*128+expert
   if getattr(self.args,'fused',False) and self.args.phase!='MEASURE':self.mismatch.logical_or_((packet[2][rows,cols]!=expert).any())
   executor=getattr(self,'graph_executor',None)
   consolidated=executor is not None and executor.wrapper and executor.mode=='replay'
   check=getattr(self,'graph_check',False) and self.index<96
   if consolidated:
    executor.gather(slot,received,rows,check);part=executor.launch(slot,len(rows),check)
   elif executor is not None:
    part=executor(slot,received[rows],check=check)
   else:
    w=self.cache[slot];part=self.kernel(received[rows],w[:1572864].view(768,2048),w[1572864:3145728].view(768,2048),w[3145728:].view(2048,768))
   self.h2d.record_slot_use(slot)
   parts[i]=executor.weight(slot,part,rw,rows,cols) if consolidated else part*rw[rows,cols,None]
  if getattr(self.args,'fused',False) and self.index>=48:return parts
  return torch.cat(parts) if parts else received.new_empty((0,2048))
 def close(self):
  if isinstance(self.h2d,PriorityH2DScheduler):self.h2d.close()
 def execute(self,*args):
  if not getattr(self.args,'physical_prefetch',False):result=super().execute(*args)
  else:
   layer,hidden,selected,weights,probs=args
   e=self.plan_event(layer,selected,weights,probs)
   dense=torch.zeros((hidden.shape[0],128),device='cuda',dtype=torch.float32).scatter_add_(1,e['targets'][selected],weights.float()).to(weights.dtype)
   fused=getattr(self.args,'fused',False) and self.index>=48
   overlap=getattr(self.args,'streaming',False) and self.index>=48
   if fused:
    before=self.transport.calls;trigger=getattr(self.args,'trigger','T1')
    if overlap:
     with nvtx_phase('moe.demand_h2d'):self.apply_fetches(e)
    if trigger=='T0':self.prefetch_next()
    with nvtx_phase('moe.forward_a2a'):pending=self.transport.forward(hidden,dense,e,async_op=True)
    if trigger=='T1':self.prefetch_next()
    with nvtx_phase('moe.forward_complete'):
     received,rw,recv_ids=pending.finish();packet=('current',received,recv_ids,rw)
    if not overlap or trigger=='T2':
     # NCCL Work.wait() orders CUDA streams; it need not block the host.
     # T2 predicts only after forward payload/unpack has physically completed.
     done=torch.cuda.Event();done.record();done.synchronize()
    if not overlap:
     with nvtx_phase('moe.demand_h2d'):self.apply_fetches(e)
    if trigger=='T2':self.prefetch_next()
   else:
    with nvtx_phase('moe.dispatch'):packet=self.dispatch(hidden,dense,e)
    done=torch.cuda.Event();done.record();done.synchronize()
    with nvtx_phase('moe.demand_h2d'):self.apply_fetches(e)
   if not overlap:
    with nvtx_phase('moe.h2d_global_barrier'):
     self.h2d.wait_slots([slot for _,_,_,slot in e['groups']],host=True)
     dist.barrier();torch.cuda.current_stream().synchronize()
   if not fused:self.prefetch_next()
   with nvtx_phase('moe.expert_compute'):values=self.compute(packet,e,layer)
   if fused:
    with nvtx_phase('moe.return_a2a'):
     precision=getattr(self.args,'partial_precision',None)
     if precision:result=combine_rank_partials(self.transport,hidden,values,e,{'bf16':torch.bfloat16,'fp32':torch.float32,'fp64':torch.float64}[precision],unique_rows=getattr(self.args,'unique_combine',False))
     else:result=self.transport.combine(hidden,values,e)
    assert self.transport.calls-before==2
   else:
    with nvtx_phase('moe.combine'):result=self.combine(hidden,values,e,packet[0])
   self.index+=1
   if self.args.phase!='MEASURE':
    self.arena.assert_consistent()
    assert np.array_equal(self.keys[self.arena.main_physical[self.rank]],self.policy.slots[self.rank,:self.cap])
    self.actual_h2d_bytes=self.h2d.metrics['bytes']
  if self.index==48:
   self.arena.assert_consistent();assert not self.arena.reservations and np.all(self.keys[self.cap:]<0)
   self.prefill_boundary=dict(main_roles=self.cap,prefetch_roles=self.args.arena_budget,prefetch_empty=True,main_resident=int(np.count_nonzero(self.keys[:self.cap]>=0)))
  return result
