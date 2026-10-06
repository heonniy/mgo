"""Strict frozen prefill plus32 decode steps; separate capture and primary timing."""
from la_physical_worker import *
from ttft_common import ROOT,GPUS,POLICIES
from mgo_v2.pinned_h2d import PriorityH2DScheduler
from mgo_v2.decode_runtime import DecodeOffloadRuntime
from mgo_v2.ready_compute import expert_order
import statistics

class GenerationRuntime(LiveRuntime):
 def __init__(self,a,model,backing,experts):
  self.capturing=a.capture
  super().__init__(a,model,backing,experts)
  self.h2d=PriorityH2DScheduler(self.cache,staging_backend='torch',cpu_team=a.staging_cpu_team);self.execution_order='streaming'
  DecodeOffloadRuntime.stage_frozen_inputs(self,0 if self.capturing else 32)
 def reset(self):
  self.policy_kind={'BR':0,'CA':1,'OLD_CA':3,'LA':4,'LA_CA_NEAR':7}[self.args.policy]
  if isinstance(getattr(self,'h2d',None),PriorityH2DScheduler):self.h2d.close();self.h2d=PriorityH2DScheduler(self.cache,staging_backend='torch',cpu_team=self.args.staging_cpu_team)
  super().reset();self.history=GateHistory(48,128,128);self.captured=[];self.post_expert_barriers=0;self.h2d_global_barriers=0;self.breakdown=[];self.layer_fetch_counts=[];self.layer_rows=[];self.ready_metrics=dict(waits=0,ready_before_first_wait=0)
 def exchange(self,x,send,recv,activation_row_bytes=0,active=False):
  y=super().exchange(x,send,recv,activation_row_bytes,active)
  if self.args.phase!='COUNTERS':
   remote=sum(send)-send[self.rank];self.actual_wire_bytes+=remote*x[0].numel()*x.element_size() if len(x) else 0
   self.actual_peer_bytes+=remote*activation_row_bytes;self.actual_collective_calls+=1
  return y
 def plan_event(self,layer,selected,weights,probs):
  if self.capturing:return self.capture_plan(layer,selected,weights,probs)
  if self.args.phase!='DIAGNOSTIC':return super().plan_event(layer,selected,weights,probs)
  original=self.policy.apply
  def measured_apply(*args,**kwargs):
   begin=time.perf_counter_ns()
   result=original(*args,**kwargs)
   self.controller_wall_ms=(time.perf_counter_ns()-begin)/1e6
   return result
  self.policy.apply=measured_apply
  try:return super().plan_event(layer,selected,weights,probs)
  finally:self.policy.apply=original
 def apply_fetches(self,e):
  for key,slot,victim,rep in e['fetches']:
   assert not rep and self.keys[slot]==victim
   self.h2d.enqueue_demand(slot,key,self.experts[key]);self.keys[slot]=key
 def compute(self,packet,e,layer):
  # Prefill ready-first execution: preserve output/group order while selecting
  # any expert whose H2D has already completed before blocking on a pending one.
  mode,received,recv_eids,rw=packet;assert mode=='current'
  groups=e['groups'];parts=[None]*len(groups)
  for i in expert_order(groups,self.h2d,True,self.ready_metrics):
   expert,rows,cols,slot=groups[i];assert self.keys[slot]==layer*128+expert
   w=self.cache[slot];gate=w[:1572864].view(768,2048);up=w[1572864:3145728].view(768,2048);down=w[3145728:].view(2048,768)
   part=self.kernel(received[rows],gate,up,down)
   self.h2d.record_slot_use(slot);parts[i]=part*rw[rows,cols,None]
  return torch.cat(parts) if parts else received.new_empty((0,2048))
 def execute(self,layer,hidden,selected,weights,probs):
  """Strict characterization substrate: no H2D/compute/communication overlap."""
  assert 0<=self.index<33*48 and layer==self.index%48
  diagnostic=self.args.phase=='DIAGNOSTIC'
  plan_start=time.perf_counter_ns() if diagnostic else None
  e=self.plan_event(layer,selected,weights,probs)
  plan_ms=(time.perf_counter_ns()-plan_start)/1e6 if diagnostic else None
  self.layer_fetch_counts.append(len(e['fetches']));self.layer_rows.append(sum(len(g[1]) for g in e['groups']))
  dense=torch.zeros((hidden.shape[0],128),device='cuda',dtype=torch.float32).scatter_add_(1,e['targets'][selected],weights.float()).to(weights.dtype)
  fetch_begin=time.perf_counter_ns() if diagnostic else None
  with nvtx_phase('moe.prefill_required_h2d'):self.apply_fetches(e)
  fetch_slots=[slot for _,slot,_,_ in e['fetches']]
  if fetch_slots:self.h2d.wait_slots(fetch_slots,host=True)
  torch.cuda.current_stream().synchronize()
  fetch_ready=time.perf_counter_ns() if diagnostic else None
  with nvtx_phase('moe.prefill_h2d_global_barrier'):
   dist.barrier();torch.cuda.current_stream().synchronize()
  fetch_aligned=time.perf_counter_ns() if diagnostic else None
  self.h2d_global_barriers+=1

  events=[torch.cuda.Event(enable_timing=True) for _ in range(5)] if diagnostic else None
  if events:events[0].record()
  with nvtx_phase('moe.prefill_forward_a2a'):packet=self.dispatch(hidden,dense,e)
  if events:events[1].record()
  with nvtx_phase('moe.prefill_expert_compute'):values=self.compute(packet,e,layer)
  if events:events[2].record()
  compute_begin=time.perf_counter_ns() if diagnostic else None
  with nvtx_phase('moe.prefill_post_expert_local_complete'):torch.cuda.current_stream().synchronize()
  compute_ready=time.perf_counter_ns() if diagnostic else None
  with nvtx_phase('moe.prefill_post_expert_global_barrier'):
   dist.barrier();torch.cuda.current_stream().synchronize()
  compute_aligned=time.perf_counter_ns() if diagnostic else None
  self.post_expert_barriers+=1
  if events:events[3].record()
  with nvtx_phase('moe.prefill_return_a2a'):result=self.combine(hidden,values,e,packet[0])
  if events:
   events[4].record();self.breakdown.append(dict(
    routing_admission_layout_ms=plan_ms,controller_wall_ms=self.controller_wall_ms,
    layer=layer,event=self.index,phase="prefill" if self.index<48 else "decode",
    h2d_local_ms=(fetch_ready-fetch_begin)/1e6,
    h2d_barrier_ms=(fetch_aligned-fetch_ready)/1e6,
    compute_local_complete_ms=(compute_ready-compute_begin)/1e6,
    compute_barrier_ms=(compute_aligned-compute_ready)/1e6,
    events=events))
  self.index+=1;return result
 def close(self):self.h2d.close()
 def forward_for(self,layer):
  rt=self
  def forward(module,hidden):
   shape=hidden.shape;flat=hidden.reshape(-1,shape[-1]);logits=module.gate(flat)
   probs=torch.softmax(logits,dim=-1,dtype=torch.float32);weights,selected=torch.topk(probs,8,dim=-1)
   weights=(weights/weights.sum(-1,keepdim=True)).to(flat.dtype)
   if not rt.capturing or rt.index<48:selected,weights=rt.route_tensors[rt.index]
   return rt.execute(layer,flat,selected,weights,probs).view(shape),logits
  return forward
 def capture_plan(self,layer,selected,weights,probs):
  g=gather_global_routes(layer,selected,weights,probs);r=g.routes;self.history.update(layer,r.full_router_probs)
  gate=self.arrays['gates'][layer] if self.index<48 else np.array([self.history.score(layer,e) for e in range(128)],np.float32)
  if self.index>=48:self.captured.append((r.selected_experts.copy(),r.routing_weights.copy(),gate.copy()))
  targets,effective,masses,lengths,destinations,fetches,row=self.policy.apply(self.index,r.selected_experts,r.routing_weights,r.origin_ranks,gate,np.zeros((128,self.world),np.int32))
  if self.index<48:assert [list(f) for f in fetches]==self.reference[self.index]
  e=plan_layout(effective,lengths,destinations,r.origin_ranks,g.counts,self.rank)
  e.update(layer=layer,targets=targets,selected=selected.cpu().numpy(),fetches=[(key,slot,victim,rep) for rank,key,slot,victim,rep in fetches if rank==self.rank])
  e['groups']=[(expert,rows,cols,int(np.flatnonzero(self.policy.slots[self.rank]==layer*128+expert)[0])) for expert,rows,cols in e['groups']]
  return device_layout(e)

HORIZON=32
METRICS=('TTFT','TPOT','E2E')
POLICY_NAMES=('BR','LA_CA_NEAR','LA')

def generation(model,rt,ids0,teacher=None):
 ids=ids0;mask=torch.ones_like(ids);past=None;tokens=[];finite=torch.ones((),dtype=torch.bool,device='cuda');step_times=[]
 dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():
  for step in range(33):
   position=torch.arange(mask.shape[1]-ids.shape[1],mask.shape[1],device='cuda')[None,:].expand(len(ids),-1)
   out=model(input_ids=ids,attention_mask=mask,position_ids=position,past_key_values=past,use_cache=True,logits_to_keep=1)
   past=out.past_key_values;next_token=out.logits[:,-1].argmax(-1);tokens.append(next_token)
   finite.logical_and_(torch.isfinite(out.logits).all());torch.cuda.synchronize();now=time.perf_counter()
   if step==0:ttft=now-start;decode_start=now
   else:step_times.append(now-previous)
   previous=now
   if step<32:
    ids=next_token[:,None] if teacher is None else teacher[:,step:step+1]
    mask=torch.cat((mask,mask.new_ones((len(ids),1))),1)
  finish=now
  actual=torch.stack(tokens).cpu().numpy();ok=bool(finite);del out,past
 assert rt.index==33*48
 return dict(TTFT=ttft,TPOT=(finish-decode_start)/32,E2E=finish-start,decode_wall=finish-decode_start,decode_steps=32,generated_outputs=33,per_step_s=step_times,argmax_hash=array_hash(actual),finite_logits=ok),actual

def validation(rt,row,rank):
 rt.h2d.synchronize();p=rt.proof
 actual=dict(state_hash=array_hash(rt.policy.slots[rank,:rt.cap]),copies=rt.h2d.metrics['copies'],bytes=rt.h2d.metrics['bytes'],canceled=rt.h2d.metrics['canceled'],barriers=rt.post_expert_barriers,h2d_barriers=rt.h2d_global_barriers,layers=rt.index,layer_fetch_counts=rt.layer_fetch_counts)
 expected=dict(state_hash=p['rank_state_hashes'][rank],copies=p['copies'][rank],bytes=p['H2D_bytes'][rank],canceled=0,barriers=1584,h2d_barriers=1584,layers=1584,layer_fetch_counts=[x[rank] for x in p['layer_fetch_counts']])
 return dict(status='PASS' if actual==expected and row['finite_logits'] else 'FAIL',observed=actual,expected=expected,prefill_H2D_copies=sum(rt.layer_fetch_counts[:48]),decode_H2D_copies=sum(rt.layer_fetch_counts[48:]),expert_rows_prefill=rt.layer_rows[:48],expert_rows_decode=rt.layer_rows[48:],wire_bytes=rt.actual_wire_bytes,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())

def freeze_inputs(source,dst,captured,teachers,capture_states):
 """CPU replay of captured common workload for all three policies."""
 from strict_headroom_common import write as atomic,sha
 import shutil
 dst.mkdir(exist_ok=True)
 receipt=json.loads((source/'receipt.json').read_text());world=receipt['world'];batch=receipt['batch'];n=world*batch
 src_s=np.load(source/'selected.npy',mmap_mode='r');src_w=np.load(source/'weights.npy',mmap_mode='r');end=len(src_s)
 ss=np.lib.format.open_memmap(dst/'selected.npy',mode='w+',shape=(end+1536*n,8),dtype='uint8');ww=np.lib.format.open_memmap(dst/'weights.npy',mode='w+',shape=ss.shape,dtype='float32')
 ss[:end]=src_s;ww[:end]=src_w
 for i,(s,w,g) in enumerate(captured):ss[end+i*n:end+(i+1)*n]=s;ww[end+i*n:end+(i+1)*n]=w
 ss.flush();ww.flush();gates=np.concatenate([np.load(source/'gates.npy'),np.array([x[2] for x in captured])]);np.save(dst/'gates.npy',gates)
 offsets=np.r_[np.load(source/'offsets.npy'),end+np.arange(1,1537)*n];np.save(dst/'offsets.npy',offsets)
 for name in ['prefill_origins.npy','decode_origins.npy','requests.json']:shutil.copyfile(source/name,dst/name)
 np.save(dst/'teacher.npy',np.concatenate(teachers,axis=1).T[:,:32]);np.save(dst/'capture_tokens.npy',np.concatenate(teachers,axis=1).T)
 origins=[np.load(dst/'prefill_origins.npy'),np.load(dst/'decode_origins.npy')];proofs={}
 for policy,kind in [('BR',0),('LA_CA_NEAR',7),('LA',4)]:
  c=Policy(receipt['capacities'],np.zeros((48,128,128),np.float32),False,kind,receipt['placement_seed']);fs=[];counts=[]
  for event in range(1584):
   lo,hi=offsets[event:event+2];out=c.apply(event,ss[lo:hi],ww[lo:hi],origins[event>=48],gates[event],np.zeros((128,world),np.int32))
   fetch=[[int(v) for v in x] for x in out[5]];assert all(x[-1]==0 for x in fetch);fs.append(fetch);counts.append(np.bincount([x[0] for x in fetch],minlength=world).tolist())
  fp=dst/f'{policy}_fetches.json';atomic(fp,fs);copies=np.array(counts).sum(0)
  proof=dict(rank_state_hashes=[array_hash(c.slots[r,:receipt['capacities'][r]]) for r in range(world)],copies=copies.tolist(),H2D_bytes=(copies*EB).tolist(),layer_fetch_counts=counts,fetches_sha256=sha(fp))
  if policy=='BR':
   for rank in range(world):assert proof['rank_state_hashes'][rank]==capture_states[rank]['state_hash'] and proof['copies'][rank]==capture_states[rank]['copies']
  proofs[policy]=proof
 receipt.update(proofs=proofs,decode_steps=32,source_inputs=str(source),capture_policy='BR',gate_history='decode W128 of captured full router probabilities; prefill existing frozen scores',files={p.name:sha(p) for p in dst.glob('*.npy')})
 atomic(dst/'receipt.json',receipt)

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);physical=list(map(int,os.environ['MGO_V2_PHYSICAL_GPUS'].split(',')));assert len(physical)==world
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 spec=json.loads(a.specs.read_text())[0];source=Path(spec['inputs']);r=json.loads((source/'receipt.json').read_text());a.inputs=source;a.capacities=r['capacities'];a.seed=r['placement_seed'];a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.output.mkdir(exist_ok=True,parents=True)
 model,backing,experts=load_model();records=json.loads((source/'requests.json').read_text())['ranks'][rank];ids=torch.tensor([x['input_ids'] for x in records],device='cuda');batch=len(ids)
 def setup(policy,phase,capture=False):
  a.policy=policy;a.phase=phase;a.capture=capture;rt=GenerationRuntime(a,model,backing,experts)
  from refactor_thread_affinity import configure
  configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt);return rt
 frozen=a.output/'frozen'
 if not (frozen/'receipt.json').exists():
  write(a.output/f'phase_rank{rank}.json',dict(stage='UNTIMED_BR_DECODE_CAPTURE'))
  rt=setup('BR','COUNTERS',True);row,tokens=generation(model,rt,ids);rt.h2d.synchronize();assert row['finite_logits']
  capture_state=dict(state_hash=array_hash(rt.policy.slots[rank,:rt.cap]),copies=rt.h2d.metrics['copies'])
  teachers=[None]*world;states=[None]*world;dist.all_gather_object(teachers,tokens);dist.all_gather_object(states,capture_state)
  if rank==0:freeze_inputs(source,frozen,rt.captured,teachers,states)
  rt.close();del rt;gc.collect();dist.barrier()
 a.inputs=frozen;teacher=torch.tensor(np.load(frozen/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda');warm={};samples={p:[] for p in POLICY_NAMES}
 for policy in POLICY_NAMES:
  write(a.output/f'phase_rank{rank}.json',dict(stage='WARM_CORRECTNESS',policy=policy));rt=setup(policy,'COUNTERS');row,tokens=generation(model,rt,ids,teacher);v=validation(rt,row,rank);assert v['status']=='PASS';warm[policy]=row['argmax_hash'];write(a.output/f'{policy}_warm_rank{rank}.json',dict(**row,validation=v));rt.close();del rt;gc.collect();dist.barrier()
 flags=[None]*world;dist.all_gather_object(flags,len(set(warm.values()))==1);assert all(flags),'cross-policy frozen32 token mismatch'
 order=list(POLICY_NAMES) if spec['order']==0 else list(reversed(POLICY_NAMES))
 for repeat in [1,2,3]:
  if repeat==3:
   delta=max(abs(samples[p][0][m]-samples[p][1][m])/statistics.mean([samples[p][0][m],samples[p][1][m]]) for p in POLICY_NAMES for m in METRICS)
   if delta<=.02 or delta>.05:break
  for policy in (order if repeat%2 else list(reversed(order))):
   rt=setup(policy,'MEASURE');gc.collect();torch.cuda.synchronize();dist.barrier();label=f'{policy}_r{repeat}'
   from torch._dynamo.utils import counters
   before=dict(counters['stats']);write(a.output/f'{label}_ready_rank{rank}.json',dict(rank=rank));dist.barrier()
   if rank==0:write(a.output/'boundary.json',dict(key=label))
   deadline=time.monotonic()+600
   while not (a.output/f'{label}_GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('release boundary')
    time.sleep(.2)
   with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generation(model,rt,ids,teacher)
   v=validation(rt,row,rank);v.update(no_compile=before==dict(counters['stats']),tokens_match_warm=row['argmax_hash']==warm[policy]);write(a.output/f'{label}_validation_rank{rank}.json',v)
   assert v['status']=='PASS' and v['no_compile'] and v['tokens_match_warm']
   write(a.output/f'{label}_measure_rank{rank}.json',dict(status='PASS',**row,validation=v));rt.close();del rt;gc.collect();dist.barrier()
   rr=[json.loads((a.output/f'{label}_measure_rank{k}.json').read_text()) for k in range(world)];samples[policy].append({m:max(x[m] for x in rr) for m in METRICS})
 if rank==0:
  estimates={p:{m:statistics.median(x[m] for x in samples[p]) for m in METRICS} for p in POLICY_NAMES}
  differences={p:{m:abs(samples[p][0][m]-samples[p][1][m])/statistics.mean([samples[p][0][m],samples[p][1][m]]) for m in METRICS} for p in POLICY_NAMES}
  unstable={p:{m:(max(x[m] for x in samples[p])-min(x[m] for x in samples[p]))/statistics.mean(x[m] for x in samples[p])>.05 for m in METRICS} for p in POLICY_NAMES}
  write(a.output/'result.json',dict(status='PASS',spec=spec,samples=samples,estimates=estimates,relative_difference=differences,unstable=unstable,gains={p:{m:1-estimates[p][m]/estimates['BR'][m] for m in METRICS} for p in POLICY_NAMES if p!='BR'},decode_steps=32,diagnostic_passes=0,scope='Frozen BR-generated tokens/routes, same policy for prefill and decode; cache carryover retained. First output from prefill plus32 subsequent decode outputs. E2E/TPOT are wall time; per-metric maximum across ranks.'))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--specs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
