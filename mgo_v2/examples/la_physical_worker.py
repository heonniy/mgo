"""R8 teacher-forced full-model BR/LA timing with live placement and fetch barrier."""
from env_offload_worker import *
from fetch_relaxed_worker import generate
from physical_repeat_rule import decide,final_unstable
import gc

class LiveRuntime(Runtime):
 def __init__(self,a,model,backing,experts):
  self.args=a;self.rank=dist.get_rank();self.world=dist.get_world_size();self.experts=experts;self.cpu_backing=backing
  self.cap=a.capacities[self.rank];self.execution_order='fetch-barrier';self.schedule_mode='live'
  self.cache=torch.empty((self.cap,EB//2),dtype=torch.bfloat16,device='cuda');self.keys=np.full(self.cap,-1,np.int32)
  self.h2d=PinnedH2DCache(self.cache,2);self.mismatch=torch.zeros((),dtype=torch.bool,device='cuda')
  self.similarity=np.zeros((48,128,128),np.float32);self.policy_kind=0 if a.policy=='BR' else 4
  if getattr(a,'live_routes',False):
   self.arrays={};self.ranges=[];self.reference=None;self.proof=None
  else:
   self.arrays={k:np.load(a.inputs/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
   self.reference=json.loads((a.inputs/(a.policy+'_fetches.json')).read_text())
   self.proof=json.loads((a.inputs/'receipt.json').read_text())['proofs'][a.policy]
   assert hashlib.sha256((a.inputs/(a.policy+'_fetches.json')).read_bytes()).hexdigest()==self.proof['fetches_sha256']
   self.ranges=[]
   for org in [self.arrays['prefill_origins'],self.arrays['decode_origins']]:
    counts=np.bincount(org,minlength=self.world);offsets=np.r_[0,np.cumsum(counts)];self.ranges.append((int(offsets[self.rank]),int(offsets[self.rank+1])))
  self.kernel=torch.compile(expert_kernel,dynamic=True,fullgraph=True);self.reset()
  for layer,block in enumerate(model.model.layers):block.mlp.forward=types.MethodType(self.forward_for(layer),block.mlp)
 def reset(self):
  self.index=0;self.keys.fill(-1);self.mismatch.zero_();self.valid=None
  self.policy=Policy(self.args.capacities,self.similarity,False,self.policy_kind,self.args.seed)
  self.actual_h2d_bytes=0;self.actual_peer_bytes=0;self.actual_wire_bytes=0;self.actual_collective_calls=0;self.actual_p2p_batches=0;self.actual_p2p_ops=0;self.actual_zero_remote_rounds=0;self.active_peer_degree_sum=0;self.max_active_peer_degree=0
 def forward_for(self,layer):
  rt=self
  def forward(module,hidden):
   shape=hidden.shape;flat=hidden.reshape(-1,shape[-1]);logits=module.gate(flat)
   probs=torch.softmax(logits,dim=-1,dtype=torch.float32);native_weights,native_selected=torch.topk(probs,8,dim=-1)
   native_weights=(native_weights/native_weights.sum(-1,keepdim=True)).to(flat.dtype)
   if rt.valid is not None:flat=flat.index_select(0,rt.valid);probs=probs[rt.valid]
   if getattr(rt.args,'live_routes',False):
    selected,weights=native_selected,native_weights
   else:
    # Fixed exact trace avoids policy-dependent route drift; router cost remains.
    lo=int(rt.arrays['offsets'][rt.index]);start,end=rt.ranges[0 if rt.index<48 else 1]
    if hasattr(rt,'route_tensors'):selected,weights=rt.route_tensors[rt.index]
    else:
     selected=torch.tensor(rt.arrays['selected'][lo+start:lo+end],device='cuda',dtype=torch.int64)
     weights=torch.tensor(rt.arrays['weights'][lo+start:lo+end],device='cuda',dtype=torch.bfloat16)
   result=rt.execute(layer,flat,selected,weights,probs)
   if rt.valid is not None:result=torch.zeros((shape[0]*shape[1],shape[2]),dtype=flat.dtype,device='cuda').index_copy_(0,rt.valid,result)
   return result.view(shape),logits
  return forward
 def plan_event(self,layer,selected,weights,probs):
  with nvtx_phase('moe.metadata_exchange'):g=gather_global_routes(layer,selected,weights,probs);r=g.routes
  with nvtx_phase('moe.rank_decision'):
   targets,effective,masses,lengths,destinations,fetches,row=self.policy.apply(self.index,r.selected_experts,r.routing_weights,r.origin_ranks,self.arrays['gates'][self.index],np.zeros((128,self.world),np.int32))
   assert [list(f) for f in fetches]==self.reference[self.index],(self.args.policy,self.index,'CPU fetch mismatch')
   self.current_global_fetch_count=len(fetches)
   e=plan_layout(effective,lengths,destinations,r.origin_ranks,g.counts,self.rank)
   e.update(layer=layer,targets=targets,selected=selected.cpu().numpy(),fetches=[(key,slot,victim,rep) for rank,key,slot,victim,rep in fetches if rank==self.rank])
   e['groups']=[(expert,rows,cols,int(np.flatnonzero(self.policy.slots[self.rank]==layer*128+expert)[0])) for expert,rows,cols in e['groups']]
  if self.args.phase!='MEASURE' and self.index%768==0:print(json.dumps(dict(policy=self.args.policy,event=self.index,phase=self.args.phase)),flush=True)
  return device_layout(e)

def main(a):
 rank=int(os.environ['RANK']);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(rank)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'));assert dist.get_world_size()==8
 receipt=json.loads((a.inputs/'receipt.json').read_text());assert receipt['status']=='PASS'
 a.seed=receipt['winner']['placement_seed'];a.capacities=[3686//8+(r<3686%8) for r in range(8)];a.comm_mode='current';a.phase='COUNTERS'
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records)
 model,backing,experts=load_model();rt=LiveRuntime(a,model,backing,experts)
 length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
 teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda')
 if rank==0:write(a.output/'phase.json',dict(stage='WARMUP_AND_VALIDATION',batch=batch,policy=a.policy))
 warm,ref_tokens=generate(model,rt,ids,mask,teacher,256)
 assert not rt.mismatch.item() and warm['state_hash']==rt.proof['rank_state_hashes'][rank]
 assert rt.actual_h2d_bytes==rt.proof['H2D_bytes'][rank]
 write(a.output/f'validation_rank{rank}.json',dict(status='PASS',H2D_bytes=rt.actual_h2d_bytes,state_hash=warm['state_hash'],argmax_hash=warm['argmax_hash'],peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
 samples=[]
 for repeat in (1,2,3):
  if repeat==3 and decide(samples)['target_repeats']!=3:break
  a.phase='MEASURE';rt.reset();torch.manual_seed(42);gc.collect();torch.cuda.synchronize();dist.barrier()
  from torch._dynamo.utils import counters
  before=dict(counters['stats'])
  write(a.output/f'ready_{repeat}_rank{rank}.json',dict(rank=rank,repeat=repeat))
  if rank==0:write(a.output/'phase.json',dict(stage='READY',repeat=repeat))
  deadline=time.monotonic()+600
  while not (a.output/f'GO_{repeat}').exists():
   if time.monotonic()>deadline:raise TimeoutError('boundary release')
   time.sleep(.2)
  with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generate(model,rt,ids,mask,teacher,256)
  assert before==dict(counters['stats']) and np.array_equal(ref_tokens,tokens)
  assert row['state_hash']==warm['state_hash'] and not rt.mismatch.item()
  row.update(status='PASS',rank=rank,batch=batch,policy=a.policy,repeat=repeat,no_compile_in_measure=True,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),affinity=cpus)
  write(a.output/f'measure_{repeat}_rank{rank}.json',row);dist.barrier()
  samples.append({k:max(json.loads((a.output/f'measure_{repeat}_rank{r}.json').read_text())[k] for r in range(8)) for k in ['E2E_wall','TPOT']})
  if rank==0:write(a.output/'phase.json',dict(stage='MEASURE_COMPLETE',repeat=repeat,samples=samples))
 if rank==0:write(a.output/'result.json',dict(status='PASS',samples=samples,gate=decide(samples[:2]),unstable=final_unstable(samples)))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',choices=['BR','LA'],required=True);main(p.parse_args())
