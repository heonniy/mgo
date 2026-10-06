"""Live-placement fixed512 prefill timing; diagnostics never primary samples."""
from la_physical_worker import *
from ttft_common import ROOT,GPUS,POLICIES
from mgo_v2.pinned_h2d import PriorityH2DScheduler
from mgo_v2.decode_runtime import DecodeOffloadRuntime
from mgo_v2.ready_compute import expert_order
import statistics

class PrefillRuntime(LiveRuntime):
 def __init__(self,a,model,backing,experts):
  super().__init__(a,model,backing,experts)
  self.h2d=PriorityH2DScheduler(self.cache,staging_backend='torch',cpu_team=a.staging_cpu_team);self.execution_order='streaming'
  DecodeOffloadRuntime.stage_frozen_inputs(self,0)
 def reset(self):
  self.policy_kind={'BR':0,'CA':1,'OLD_CA':3,'LA':4,'LA_CA_NEAR':7}[self.args.policy]
  if isinstance(getattr(self,'h2d',None),PriorityH2DScheduler):self.h2d.close();self.h2d=PriorityH2DScheduler(self.cache,staging_backend='torch',cpu_team=self.args.staging_cpu_team)
  super().reset();self.post_expert_barriers=0;self.h2d_global_barriers=0;self.breakdown=[];self.layer_fetch_counts=[];self.layer_rows=[];self.ready_metrics=dict(waits=0,ready_before_first_wait=0)
 def exchange(self,x,send,recv,activation_row_bytes=0,active=False):
  y=super().exchange(x,send,recv,activation_row_bytes,active)
  if self.args.phase!='COUNTERS':
   remote=sum(send)-send[self.rank];self.actual_wire_bytes+=remote*x[0].numel()*x.element_size() if len(x) else 0
   self.actual_peer_bytes+=remote*activation_row_bytes;self.actual_collective_calls+=1
  return y
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
  assert 0<=self.index<48 and layer==self.index
  diagnostic=self.args.phase=='DIAGNOSTIC';e=self.plan_event(layer,selected,weights,probs)
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

  events=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if diagnostic else None
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
  with nvtx_phase('moe.prefill_return_a2a'):result=self.combine(hidden,values,e,packet[0])
  if events:
   events[3].record();self.breakdown.append(dict(
    layer=layer,
    h2d_local_ms=(fetch_ready-fetch_begin)/1e6,
    h2d_barrier_ms=(fetch_aligned-fetch_ready)/1e6,
    compute_local_complete_ms=(compute_ready-compute_begin)/1e6,
    compute_barrier_ms=(compute_aligned-compute_ready)/1e6,
    events=events))
  self.index+=1;return result
 def close(self):self.h2d.close()

def run_prefill(model,rt,ids,mask):
 # Align ranks after file-based release; polling latency is outside TTFT.
 context=ids.shape[1];dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():
  out=model(input_ids=ids,attention_mask=mask,position_ids=torch.arange(context,device='cuda')[None,:].expand(len(ids),-1),use_cache=True,logits_to_keep=1)
  prefill=time.perf_counter();tokens=out.logits[:,-1].argmax(-1);actual=tokens.cpu().numpy();torch.cuda.synchronize();ttft=time.perf_counter()-start
  finite=bool(torch.isfinite(out.logits).all());del out,tokens
 return dict(TTFT=ttft,prefill_host_submit_s=prefill-start,argmax_hash=array_hash(actual),first_tokens=actual.tolist(),finite_logits=finite)

def check(rt,row,rank):
 rt.h2d.synchronize();proof=rt.proof
 observed=dict(state_hash=array_hash(rt.policy.slots[rank,:rt.cap]),copies=rt.h2d.metrics['copies'],bytes=rt.h2d.metrics['bytes'],canceled=rt.h2d.metrics['canceled'],barriers=rt.post_expert_barriers,h2d_barriers=rt.h2d_global_barriers,layers=rt.index,layer_fetch_counts=rt.layer_fetch_counts)
 expected=dict(state_hash=proof['rank_state_hashes'][rank],copies=proof['copies'][rank],bytes=proof['H2D_bytes'][rank],canceled=0,barriers=48,h2d_barriers=48,layers=48,layer_fetch_counts=[x[rank] for x in proof['layer_fetch_counts']])
 return dict(status='PASS' if observed==expected and row['finite_logits'] else 'FAIL',observed=observed,expected=expected,expert_rows=rt.layer_rows,wire_bytes=rt.actual_wire_bytes,peer_bytes=rt.actual_peer_bytes,collective_calls=rt.actual_collective_calls,ready_metrics=dict(rt.ready_metrics),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);physical=[int(x) for x in os.environ.get('MGO_V2_PHYSICAL_GPUS',','.join(map(str,GPUS))).split(',')];assert len(physical)==world;gpu=physical[rank];cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(gpu)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 a.output.mkdir(parents=True,exist_ok=True);specs=json.loads(a.specs.read_text());model,backing,experts=load_model();results=[]
 for spec in specs:
  key=spec['key'];a.inputs=Path(spec['inputs']);source=json.loads((a.inputs/'receipt.json').read_text());a.capacities=source['capacities'];a.seed=source['placement_seed'];a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.phase='COUNTERS'
  receipt=json.loads((a.inputs/'receipt.json').read_text());context=int(receipt['context']);records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];ids=torch.tensor([r['input_ids'] for r in records],device='cuda');assert ids.shape[1]==context;mask=torch.ones_like(ids);assert bool((mask.sum(1)==context).all())
  candidate=spec['policy'];order=['BR',candidate] if spec['order']==0 else [candidate,'BR'];warm={};samples={p:[] for p in order};rts={}
  # Keep one arena alive at a time; policies do not share mutable cache state.
  def setup(policy,phase):
   a.policy=policy;a.phase=phase
   rt=PrefillRuntime(a,model,backing,experts)
   from refactor_thread_affinity import configure
   rt.thread_placement=configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt)
   return rt
  for policy in order:
   rt=setup(policy,'COUNTERS');write(a.output/f'phase_rank{rank}.json',dict(stage='WARM_CORRECTNESS',key=key,policy=policy))
   row=run_prefill(model,rt,ids,mask);validation=check(rt,row,rank);warm[policy]=row
   write(a.output/f'{key}_{policy}_warm_rank{rank}.json',dict(**row,validation=validation));rt.close();del rt;gc.collect();dist.barrier()
   assert all(json.loads((a.output/f'{key}_{policy}_warm_rank{r}.json').read_text())['validation']['status']=='PASS' for r in range(4))
  same_tokens=warm['BR']['argmax_hash']==warm[candidate]['argmax_hash']
  flags=[None]*4;dist.all_gather_object(flags,same_tokens);valid=all(flags)
  write(a.output/f'{key}_pair_gate_rank{rank}.json',dict(status='PASS' if valid else 'INVALID',tokens_match=flags,scope='Cross-policy first-token parity on frozen routing; no substitution or replica changes.'))
  if not valid:
   if rank==0:write(a.output/f'{key}_result.json',dict(status='INVALID',reason='cross-policy token mismatch',spec=spec))
   continue
  for repeat in (1,2,3):
   if a.stage=='S1' and repeat>1:break
   current=order if repeat%2 else list(reversed(order))
   needed=current
   if repeat==3:
    diff=max(abs(samples[p][0]-samples[p][1])/statistics.mean(samples[p][:2]) for p in order)
    if diff<=.02 or diff>.05:break
   for policy in needed:
    rt=setup(policy,'MEASURE');torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
    from torch._dynamo.utils import counters
    before=dict(counters['stats']);label=f'{key}_{policy}_r{repeat}'
    write(a.output/f'{label}_ready_rank{rank}.json',dict(rank=rank,key=label));dist.barrier()
    if rank==0:write(a.output/'boundary.json',dict(key=label))
    deadline=time.monotonic()+600
    while not (a.output/f'{label}_GO').exists():
     if time.monotonic()>deadline:raise TimeoutError('TTFT clean boundary')
     time.sleep(.2)
    with torch._dynamo.config.patch(error_on_recompile=True):row=run_prefill(model,rt,ids,mask)
    validation=check(rt,row,rank);compile_ok=before==dict(counters['stats']);tokens_ok=row['argmax_hash']==warm[policy]['argmax_hash']
    write(a.output/f'{label}_validation_rank{rank}.json',dict(**validation,no_compile=compile_ok,tokens_match_warm=tokens_ok))
    assert validation['status']=='PASS' and compile_ok and tokens_ok
    write(a.output/f'{label}_measure_rank{rank}.json',dict(status='PASS',rank=rank,policy=policy,repeat=repeat,**row,validation=validation));rt.close();del rt;gc.collect();dist.barrier()
    samples[policy].append(max(json.loads((a.output/f'{label}_measure_rank{r}.json').read_text())['TTFT'] for r in range(4)))
  if a.stage=='S2':
   for policy in order:
    rt=setup(policy,'DIAGNOSTIC');row=run_prefill(model,rt,ids,mask);validation=check(rt,row,rank)
    phases=[]
    for x in rt.breakdown:
     e=x.pop('events');phases.append(dict(**x,forward_cuda_ms=e[0].elapsed_time(e[1]),expert_cuda_ms=e[1].elapsed_time(e[2]),return_cuda_ms=e[2].elapsed_time(e[3])))
    calibration=[]
    for i in range(120):
     t=time.perf_counter_ns();dist.barrier();torch.cuda.current_stream().synchronize()
     if i>=20:calibration.append((time.perf_counter_ns()-t)/1e6)
    write(a.output/f'{key}_{policy}_diagnostic_rank{rank}.json',dict(status=validation['status'],primary_timing=False,validation=validation,phases=phases,idle_barrier_reference_ms=statistics.median(calibration),first_tokens=row['first_tokens'],argmax_hash=row['argmax_hash']))
    assert validation['status']=='PASS' and row['argmax_hash']==warm[policy]['argmax_hash']
    rt.close();del rt;gc.collect();dist.barrier()
  if rank==0:
   delta={p:None if len(v)<2 else abs(v[0]-v[1])/statistics.mean(v[:2]) for p,v in samples.items()};est={p:statistics.median(v) for p,v in samples.items()};unstable=any(len(v)>1 and (max(v)-min(v))/statistics.mean(v)>.05 for v in samples.values())
   result=dict(status='PASS',spec=spec,samples=samples,estimates=est,relative_difference=delta,unstable=unstable,selection_only=a.stage=='S1',gain=1-est[candidate]/est['BR']);write(a.output/f'{key}_result.json',result);results.append(result)
  dist.barrier()
 if rank==0:write(a.output/'result.json',dict(status='PASS',stage=a.stage,results=results))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--specs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['S1','S2'],required=True);main(p.parse_args())
