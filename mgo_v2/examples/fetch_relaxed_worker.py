"""Resident fixed-route, teacher-forced A3/A2 decode64 physical screen."""
from env_offload_worker import *
from env_offload_tensors import pack_layouts

class FrozenRuntime:
 def __init__(self,experts,backing,plan,policy,rank,world):
  self.rank=rank;self.world=world;self.experts=experts;self.cpu_backing=backing;self.plan=plan;self.policy=policy;self.proof=plan['proofs'][policy][rank];self.cap=self.proof['capacity']
  path=Path(plan['path'])/f'{policy}_rank{rank}.pkl.gz';assert hashlib.sha256(path.read_bytes()).hexdigest()==self.proof['schedule_sha256']
  with gzip.open(path,'rb') as f:raw=pickle.load(f)
  self.events=pack_layouts(raw)
  for e in self.events:
   e['weights']=torch.as_tensor(e['weights'],device='cuda',dtype=torch.bfloat16);e['return_weights']=torch.as_tensor(e['return_weights'],device='cuda',dtype=torch.bfloat16)
  self.cache=torch.empty((self.cap,EB//2),device='cuda',dtype=torch.bfloat16);self.keys=np.full(self.cap,-1,np.int32);self.kernel=torch.compile(expert_kernel,dynamic=True,fullgraph=True);self.reference_parts={};self.reference_tokens=None;self.reset()
 def attach(self,model):
  for l,block in enumerate(model.model.layers):block.mlp.forward=types.MethodType(self.forward(l),block.mlp)
 def reset(self):
  self.keys.fill(-1);self.index=0;self.valid=None;self.calls=0;self.h2d=0;self.bytes=[0,0,0];self.max_abs=0.;self.max_rel=0.;self.mode='UNTIMED';self.variant='A3';self.traffic=[]
 def forward(self,layer):
  rt=self
  def forward(module,hidden):
   shape=hidden.shape;flat=hidden.reshape(-1,shape[-1]);logits=module.gate(flat)
   # Execute the native router, but consume only common frozen IDs/weights.
   probs=torch.softmax(logits,dim=-1,dtype=torch.float32);weight,selected=torch.topk(probs,8,dim=-1);weight=(weight/weight.sum(-1,keepdim=True)).to(flat.dtype)
   if rt.valid is not None:flat=flat.index_select(0,rt.valid)
   result=rt.execute(layer,flat)
   if rt.valid is not None:result=torch.zeros((shape[0]*shape[1],shape[2]),dtype=flat.dtype,device='cuda').index_copy_(0,rt.valid,result)
   return result.view(shape),logits
  return forward
 def exchange(self,x,send,recv,kind):
  if self.mode=='COUNTERS':
   self.calls+=1;self.bytes[kind]+=(sum(send)-send[self.rank])*x.shape[-1]*x.element_size()
   if kind!=1:self.traffic.append([(sum(send)-send[self.rank])*4096,(sum(recv)-recv[self.rank])*4096])
  y=torch.empty((sum(recv),)+tuple(x.shape[1:]),device='cuda',dtype=x.dtype);dist.all_to_all_single(y,x.contiguous(),output_split_sizes=recv,input_split_sizes=send);return y
 def execute(self,layer,hidden):
  e=self.events[self.index];assert e['layer']==layer
  for key,slot,victim,rep in e['fetches']:
   assert not rep and self.keys[slot]==victim;pos=0
   for tensor in self.experts[key]:
    n=tensor.numel();self.cache[slot,pos:pos+n].copy_(tensor.reshape(-1),non_blocking=True);pos+=n
   assert pos*2==EB;self.keys[slot]=key
   if self.mode=='COUNTERS':self.h2d+=EB
  received=self.exchange(hidden[e['send_idx']],e['send_counts'],e['recv_counts'],0)
  if self.variant=='A3':
   dense=torch.zeros((hidden.shape[0],128),device='cuda',dtype=torch.float32).scatter_add_(1,e['selected'],e['weights'].float()).to(hidden.dtype)
   send_w=dense[e['send_idx'][:,None],e['send_eids'].clamp_min(0)]*(e['send_eids']>=0)
   rw=self.exchange(send_w,e['send_counts'],e['recv_counts'],1)
  parts=[]
  for expert,rows,cols,slot in e['groups']:
   assert self.keys[slot]==layer*128+expert;weight=self.cache[slot]
   value=self.kernel(received[rows],weight[:1572864].view(768,2048),weight[1572864:3145728].view(768,2048),weight[3145728:].view(2048,768))
   if self.variant=='A3':value=value*rw[rows,cols,None]
   parts.append(value)
  values=torch.cat(parts) if parts else hidden.new_empty((0,2048));returned=self.exchange(values[e['return_order']],e['return_counts'],e['return_recv_counts'],2)
  if self.variant=='A2':returned=returned*e['return_weights'][:,None]
  if self.mode=='COUNTERS' and 48<=self.index<96:
   if self.variant=='A3':self.reference_parts[self.index]=returned.detach().clone()
   else:
    ref=self.reference_parts[self.index];delta=(returned.float()-ref.float()).abs();self.max_abs=max(self.max_abs,float(delta.max()) if delta.numel() else 0.);self.max_rel=max(self.max_rel,float((delta/ref.float().abs().clamp_min(1e-6)).max()) if delta.numel() else 0.)
  output=torch.zeros_like(hidden)
  for idx,pos in e['combine']:output.index_add_(0,idx,returned[pos])
  self.index+=1;return output

def generate(model,rt,ids0,mask0,teacher,horizon=64):
 check_cpu_residency(rt.cpu_backing);ids=ids0;mask=mask0;past=None;rt.valid=mask0.reshape(-1).nonzero().flatten();argmaxes=torch.empty((horizon+1,ids.shape[0]),dtype=torch.long,device='cuda')
 dist.barrier();torch.cuda.synchronize();start=time.perf_counter();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 with torch.inference_mode():
  for step in range(horizon+1):
   position=mask.cumsum(-1)-1;position.masked_fill_(mask==0,0)
   out=model(input_ids=ids,attention_mask=mask,position_ids=position[:,-ids.shape[1]:],past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
   logits=out.logits[:,-1].clone();logits[:,model.generation_config.eos_token_id]=-torch.inf;argmaxes[step]=logits.argmax(-1)
   if step<horizon:ids=teacher[:,step:step+1];mask=torch.cat((mask,mask.new_ones((mask.shape[0],1))),1)
   if step==0:torch.cuda.synchronize();decode_start=time.perf_counter();begin.record();rt.valid=None
  end.record();torch.cuda.synchronize();finish=time.perf_counter()
 dist.barrier();tokens=argmaxes.cpu().numpy();assert rt.index==(horizon+1)*48
 return dict(E2E_wall=finish-start,decode_wall=finish-decode_start,TPOT=begin.elapsed_time(end)/1000/horizon,argmax_hash=array_hash(tokens),teacher_hash=array_hash(teacher[:,:horizon].cpu().numpy()),state_hash=array_hash(rt.keys)),tokens

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);cpus=list(range(24*rank,24*(rank+1)))
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'));model,backing,experts=load_model();plans=[p for p in json.loads(a.plans.read_text()) if p['world']==world];contexts={}
 for plan in plans:
  path=Path(plan['path']);records=json.loads((path/'requests.json').read_text())['ranks'][rank];teacher=torch.from_numpy(np.load(path/'teacher_tokens.npy')[rank*32:(rank+1)*32]).to('cuda');length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
  ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
  for policy in ['BR','CA']:contexts[plan['id'],policy]=(FrozenRuntime(experts,backing,plan,policy,rank,world),ids,mask,teacher)
 first=next(iter(contexts.values()))
 for variant in ['A3','A2']:
  rt,ids,mask,teacher=first;rt.reset();rt.variant=variant;rt.attach(model)
  if rank==0:write(a.output/'phase.json',dict(status='SHORT_READINESS',runtime=variant,decode_steps=8))
  generate(model,rt,ids,mask,teacher,8)
 # Required full correctness/COUNTERS passes; no full warmup before MEASURE.
 for (key,policy),(rt,ids,mask,teacher) in contexts.items():
  for variant in ['A3','A2']:
   if rank==0:write(a.output/'phase.json',dict(status='COUNTERS_AND_CORRECTNESS',workload=key,policy=policy,runtime=variant))
   rt.reset();rt.variant=variant;rt.mode='COUNTERS';rt.attach(model);receipt,tokens=generate(model,rt,ids,mask,teacher)
   assert receipt['state_hash']==rt.proof['state_hash'] and rt.h2d==rt.proof['H2D_bytes']
   assert rt.bytes==[rt.proof['dispatch_bytes'],rt.proof['weight_bytes'] if variant=='A3' else 0,rt.proof['return_bytes']]
   assert rt.calls==(9360 if variant=='A3' else 6240)
   if variant=='A3':rt.reference_tokens=tokens
   else:assert np.array_equal(tokens,rt.reference_tokens),'A2 full64 argmax differs'
   receipt.update(status='PASS',phase='COUNTERS',world=world,rank=rank,workload=key,policy=policy,variant=variant,H2D_bytes=rt.h2d,activation_bytes=rt.bytes[0],weight_bytes=rt.bytes[1],return_bytes=rt.bytes[2],a2a_calls=rt.calls,max_abs_weighted_partial=rt.max_abs,max_rel_weighted_partial=rt.max_rel,weighted_partial_prefix='first decode forward, all48 layers',route_weight_sha256=rt.proof['route_weight_sha256'],schedule_sha256=rt.proof['schedule_sha256'])
   np.save(a.output/f'{key}_{policy}_{variant}_traffic_rank{rank}.npy',np.array(rt.traffic,np.int64));write(a.output/f'{key}_{policy}_{variant}_COUNTERS_rank{rank}.json',receipt)
  rt.reference_parts.clear()
 dist.barrier()
 def measure(key,policy,variant,repeat):
  rt,ids,mask,teacher=contexts[key,policy];rt.reset();rt.variant=variant;rt.mode='MEASURE';rt.attach(model);torch.manual_seed(42)
  from torch._dynamo.utils import counters
  before=dict(counters['stats']);label=f'{key}_{policy}_{variant}_{repeat}'
  write(a.output/f'ready_{label}_rank{rank}.json',dict(status='READY',rank=rank,label=label));dist.barrier()
  if rank==0:write(a.output/'phase.json',dict(status='READY',label=label))
  deadline=time.monotonic()+600
  while not (a.output/('GO_'+label)).exists():
   if time.monotonic()>deadline:raise TimeoutError('coordinator boundary timeout')
   time.sleep(.1)
  with torch._dynamo.config.patch(error_on_recompile=True):receipt,tokens=generate(model,rt,ids,mask,teacher)
  assert dict(counters['stats'])==before and np.array_equal(tokens,rt.reference_tokens) and receipt['state_hash']==rt.proof['state_hash']
  receipt.update(status='PASS',phase='MEASURE',world=world,rank=rank,workload=key,policy=policy,variant=variant,repeat=repeat,no_compile_in_measure=True,affinity=cpus,boot=BOOT,route_weight_sha256=rt.proof['route_weight_sha256'],schedule_sha256=rt.proof['schedule_sha256'])
  write(a.output/f'{label}_rank{rank}.json',receipt);dist.barrier()
  return {k:max(json.loads((a.output/f'{label}_rank{r}.json').read_text())[k] for r in range(world)) for k in ['E2E_wall','decode_wall','TPOT']}
 for plan in plans:
  key=plan['id']
  for variant in ['A3','A2']:
   br=measure(key,'BR',variant,1);ca=measure(key,'CA',variant,1);gains={k:1-ca[k]/br[k] for k in br};confirm=max(gains.values())>=.01
   if rank==0:write(a.output/f'{key}_{variant}_confirmation.json',dict(gains=gains,confirm=confirm,threshold=.01,max_confirmation_pairs=1))
   if confirm:
    measure(key,'BR',variant,2);measure(key,'CA',variant,2)
 if rank==0:write(a.output/'phase.json',dict(status='COMPLETE'))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--plans',type=Path,required=True);main(p.parse_args())
