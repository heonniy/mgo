"""Physical frozen-plan Qwen offloading; separate planning and clean execution."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TOKENIZERS_PARALLELISM']='false'
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import()
import argparse,ctypes,gzip,hashlib,json,pickle,struct,time,types
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig,AutoModelForCausalLM,GenerationConfig
from mgo_v2.model_loader import checkpoint_identity,prepare_checkpoint_store,EXPERT
from mgo_v2.eviction import GateHistory
from mgo_v2.runtime import gather_global_routes
from mgo_v2.pinned_h2d import PinnedH2DCache
from mgo_v2.active_peer import exchange_active_peers
from env_offload_policy import Policy
from br_carep_cpu import suffix_demand
PKG=Path(__file__).resolve().parents[1]
PACKET=PKG/'experiments/env_e2e_tpot_offload_20261003'
ROOT=Path('/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003')
SOURCE=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
MODEL='/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
STORE=Path('/home/hwlee/mgo-results/runtime_validation_20261001/expert_store')
EB=9437184

def digest(x):return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
def array_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def check_cpu_residency(backing):
 page=os.sysconf('SC_PAGE_SIZE');count=(backing.numel()+page-1)//page
 flags=np.empty(count,np.uint8);lib=ctypes.CDLL(None,use_errno=True)
 rc=lib.mincore(ctypes.c_void_p(backing.data_ptr()),ctypes.c_size_t(backing.numel()),ctypes.c_void_p(flags.ctypes.data))
 if rc:raise OSError(ctypes.get_errno(),'mincore expert pool')
 assert np.all(flags&1),'CPU expert pool contains nonresident pages'
 return count

def expert_kernel(x,gate,up,down):return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)

def load_model():
 manifest=prepare_checkpoint_store(MODEL,STORE)
 config=AutoConfig.from_pretrained(MODEL,local_files_only=True);config._attn_implementation='sdpa'
 with init_empty_weights():model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16)
 _,wm=checkpoint_identity(MODEL)
 for shard in sorted(set(wm.values())):
  names=[k for k,v in wm.items() if v==shard and not EXPERT.fullmatch(k)]
  if names:
   with safe_open(str(Path(MODEL)/shard),framework='pt',device='cpu') as f:
    for name in names:set_module_tensor_to_device(model,name,'cuda:0',value=f.get_tensor(name),dtype=torch.bfloat16)
 for module in model.modules():
  if module.__class__.__name__=='Qwen3MoeRotaryEmbedding':
   inv,module.attention_scaling=module.rope_init_fn(config,device=torch.device('cuda:0'));module.register_buffer('inv_freq',inv,persistent=False);module.original_inv_freq=inv
 # Read-only use of a shared file-backed CPU pool; fault all pages in before execution.
 backing=torch.from_file(str(STORE/'archer_param_0'),shared=False,size=manifest['bytes'],dtype=torch.uint8)
 _=backing[::4096].to(torch.int64).sum().item()  # untimed CPU residency preflight
 index={}
 with (STORE/'archer_index').open('rb') as f:
  count=struct.unpack('<I',f.read(4))[0]
  for _ in range(count):
   tid,file_id,offset,size,ndim=struct.unpack('<IIqQq',f.read(32));shape=struct.unpack('<'+'q'*ndim,f.read(8*ndim));f.read(6)
   assert file_id==0;index[tid]=(offset,size,shape)
 experts=[]
 for key in range(48*128):
  tensors=[]
  for i in range(3):
   offset,size,shape=index[key*3+i];tensors.append(backing[offset:offset+size].view(torch.bfloat16).view(shape))
  experts.append(tensors)
 assert all(sum(t.numel()*2 for t in group)==EB and all(t.device.type=='cpu' for t in group) for group in experts)
 model.generation_config=GenerationConfig.from_pretrained(MODEL,local_files_only=True)
 model.eval();return model,backing,experts

from env_offload_layout import add_coslot_layout,plan_layout

from env_offload_tensors import pack_layouts

def device_layout(e):return pack_layouts([e])[0]

class Runtime:
 def __init__(self,args,model,experts,cap):
  self.args=args;self.rank=dist.get_rank();self.world=dist.get_world_size();self.experts=experts;self.cap=cap;self.events=[];self.metrics=[];self.index=0
  self.cache=torch.empty((cap,EB//2),dtype=torch.bfloat16,device='cuda');self.keys=np.full(cap,-1,np.int32);self.mismatch=torch.zeros((),dtype=torch.bool,device='cuda')
  self.h2d=PinnedH2DCache(self.cache,args.h2d_stages) if args.h2d_mode=='pinned' else None
  self.history=GateHistory(48,128,128);self.valid=None;self.actual_h2d_bytes=0;self.actual_peer_bytes=0;self.actual_wire_bytes=0;self.actual_collective_calls=0;self.actual_p2p_batches=0;self.actual_p2p_ops=0;self.actual_zero_remote_rounds=0;self.active_peer_degree_sum=0;self.max_active_peer_degree=0
  if args.phase=='PLAN' or (args.phase=='COUNTERS' and args.comm_mode=='current'):
   meta=json.loads((PKG/'experiments/br_ca_carep_cpu_headroom_20261003/similarity_audit.json').read_text());assert hashlib.sha256(Path(meta['similarity_path']).read_bytes()).hexdigest()==meta['similarity_sha256'];sim=np.load(meta['similarity_path'])
   self.policy=Policy(args.capacities,sim,args.substitution,['BR','CA','CA-rep'].index(args.policy))
   self.future=None
   if args.policy=='CA-rep':
    path=SOURCE/'packed'/args.dataset/f'R{self.world}_B{args.batch}'
    self.future=suffix_demand(*(np.load(path/(k+'.npy'),mmap_mode='r') for k in ['selected','offsets','decode_origins']),48,128,self.world,256)
  if args.phase!='PLAN':
   validation=json.loads((args.plan/'schedule_validation.json').read_text());assert validation['status']=='PASS'
   self.plan_proof=validation['ranks'][self.rank]
   assert hashlib.sha256((args.plan/f'rank{self.rank}.pkl.gz').read_bytes()).hexdigest()==self.plan_proof['schedule_file_sha256']
   with gzip.open(args.plan/f'rank{self.rank}.pkl.gz','rb') as f:self.events=pickle.load(f)
   if args.comm_mode in ('coslot','coslot-active'):self.events=[add_coslot_layout(e,self.world) for e in self.events]
   self.events=pack_layouts(self.events)
   if args.phase=='COUNTERS' and args.comm_mode in ('coslot','coslot-active'):self.metrics=np.load(args.plan/f'rank{self.rank}_metrics.npy').tolist()
  self.kernel=torch.compile(expert_kernel,dynamic=True,fullgraph=True)
  for l,block in enumerate(model.model.layers):
   block.mlp.forward=types.MethodType(self.forward_for(l),block.mlp)
 def reset(self):self.index=0;self.keys.fill(-1);self.mismatch.zero_()
 def forward_for(self,layer):
  rt=self
  def forward(module,hidden_states):
   shape=hidden_states.shape;flat=hidden_states.reshape(-1,shape[-1]);logits=module.gate(flat)
   probs=torch.softmax(logits,dim=-1,dtype=torch.float32);weights,selected=torch.topk(probs,8,dim=-1);weights=(weights/weights.sum(-1,keepdim=True)).to(flat.dtype)
   valid=rt.valid
   if valid is not None:flat=flat.index_select(0,valid);selected=selected[valid];weights=weights[valid];probs=probs[valid]
   result=rt.execute(layer,flat,selected,weights,probs)
   if valid is not None:result=torch.zeros((shape[0]*shape[1],shape[2]),dtype=flat.dtype,device='cuda').index_copy_(0,valid,result)
   return result.view(shape),logits
  return forward
 def exchange(self,x,send,recv,activation_row_bytes=0,active=False):
  row_bytes=x.element_size()
  for dim in x.shape[1:]:row_bytes*=dim
  remote=sum(send)-send[self.rank]
  if self.args.phase=='COUNTERS':
   self.actual_wire_bytes+=remote*row_bytes
   if activation_row_bytes:self.actual_peer_bytes+=remote*activation_row_bytes
  if active:
   y,peers,nops=exchange_active_peers(x,send,recv)
   if self.args.phase=='COUNTERS':
    degree=len(peers);self.active_peer_degree_sum+=degree;self.max_active_peer_degree=max(self.max_active_peer_degree,degree)
    if nops:self.actual_p2p_batches+=1;self.actual_p2p_ops+=nops
    else:self.actual_zero_remote_rounds+=1
   return y
  if self.args.phase=='COUNTERS':self.actual_collective_calls+=1
  y=torch.empty((sum(recv),)+tuple(x.shape[1:]),dtype=x.dtype,device='cuda');dist.all_to_all_single(y,x.contiguous(),output_split_sizes=recv,input_split_sizes=send);return y
 def exchange_coslot_routes(self,hidden,dense,e):
  idx=e['coslot_send_idx'];eids=e['coslot_send_eids'];send_hidden=hidden.index_select(0,idx).contiguous();send_weights=dense[idx,eids].contiguous()
  k=send_hidden.shape[0];h=hidden.shape[-1];eh=hidden.element_size();ee=eids.element_size();ew=send_weights.element_size();hb=h*eh;row=hb+ee+ew
  send_buf=torch.empty((k,row),dtype=torch.uint8,device='cuda')
  send_buf[:,:hb]=send_hidden.view(torch.uint8).view(k,hb)
  send_buf[:,hb:hb+ee]=eids.contiguous().reshape(k,1).view(torch.uint8)
  send_buf[:,hb+ee:]=send_weights.reshape(k,1).view(torch.uint8)
  recv_buf=self.exchange(send_buf,e['coslot_send_counts'],e['coslot_recv_counts'],activation_row_bytes=hb,active=self.args.comm_mode=='coslot-active')
  recv_k=recv_buf.shape[0]
  recv_hidden=torch.empty((recv_k,h),dtype=hidden.dtype,device='cuda');recv_hidden.view(torch.uint8).view(recv_k,hb).copy_(recv_buf[:,:hb])
  recv_eids=torch.empty(recv_k,dtype=eids.dtype,device='cuda');recv_eids.view(torch.uint8).view(recv_k,ee).copy_(recv_buf[:,hb:hb+ee])
  recv_weights=torch.empty(recv_k,dtype=send_weights.dtype,device='cuda');recv_weights.view(torch.uint8).view(recv_k,ew).copy_(recv_buf[:,hb+ee:])
  return recv_hidden,recv_eids,recv_weights
 def execute(self,layer,hidden,selected,weights,probs):
  if self.args.phase=='PLAN' or (self.args.phase=='COUNTERS' and self.args.comm_mode=='current'):
   g=gather_global_routes(layer,selected,weights,probs);r=g.routes;self.history.update(layer,r.full_router_probs)
   scores=np.array([self.history.score(layer,e) for e in range(128)],np.float32)
   future=self.future[self.index//48,layer] if self.future is not None else np.zeros((128,self.world),np.int32)
   targets,effective,masses,lengths,destinations,fetches,row=self.policy.apply(self.index,r.selected_experts,r.routing_weights,r.origin_ranks,scores,future)
   e=plan_layout(effective,lengths,destinations,r.origin_ranks,g.counts,self.rank)
   e.update(layer=layer,targets=targets,selected=selected.cpu().numpy(),fetches=[(key,slot,victim,rep) for rank,key,slot,victim,rep in fetches if rank==self.rank])
   e['groups']=[(expert,rows,cols,int(np.flatnonzero(self.policy.slots[self.rank]==layer*128+expert)[0])) for expert,rows,cols in e['groups']]
   if self.args.comm_mode in ('coslot','coslot-active'):e=add_coslot_layout(e,self.world)
   if self.args.phase=='PLAN':self.events.append(e)
   else:
    frozen=self.events[self.index]
    assert e['fetches']==frozen['fetches'] and e['send_counts']==frozen['send_counts'] and e['return_counts']==frozen['return_counts']
    assert np.array_equal(e['targets'],frozen['targets'].cpu().numpy())
   self.metrics.append(row);e=device_layout(e) if self.args.phase=='PLAN' else self.events[self.index]
  else:e=self.events[self.index]
  assert e['layer']==layer
  self.mismatch.logical_or_((selected!=e['selected']).any())
  # CoSLoT-style pinned mode: bounded pinned stage -> dedicated H2D stream.
  fetch_slots=[slot for _,slot,_,_ in e['fetches']];assert len(fetch_slots)==len(set(fetch_slots)),'slot fetched twice before compute'
  for key,slot,victim,rep in e['fetches']:
   assert self.keys[slot]==victim
   if self.h2d:self.h2d.enqueue(slot,self.experts[key]);pos=EB//2
   else:
    pos=0
    for t in self.experts[key]:
     n=t.numel();self.cache[slot,pos:pos+n].copy_(t.reshape(-1),non_blocking=True);pos+=n
   assert pos*2==EB;self.keys[slot]=key
   if self.args.phase=='COUNTERS':self.actual_h2d_bytes+=pos*2
  dense=torch.zeros((hidden.shape[0],128),device='cuda',dtype=torch.float32).scatter_add_(1,e['targets'][selected],weights.float()).to(weights.dtype)
  if self.args.comm_mode in ('coslot','coslot-active'):
   received,recv_eids,rw=self.exchange_coslot_routes(hidden,dense,e);parts=[]
   for expert,rows,slot in e['coslot_groups']:
    assert self.keys[slot]==layer*128+expert
    if self.args.phase!='MEASURE':self.mismatch.logical_or_((recv_eids[rows]!=expert).any())
    if self.h2d:self.h2d.wait_for_slot(slot)
    w=self.cache[slot];gate=w[:1572864].view(768,2048);up=w[1572864:3145728].view(768,2048);down=w[3145728:].view(2048,768)
    part=self.kernel(received[rows],gate,up,down)
    if self.h2d:self.h2d.record_slot_use(slot)
    parts.append(part*rw[rows,None])
   values=torch.cat(parts) if parts else hidden.new_empty((0,2048))
   returned=self.exchange(values[e['coslot_return_order']],e['coslot_return_counts'],e['coslot_return_recv_counts'],activation_row_bytes=2048*values.element_size(),active=self.args.comm_mode=='coslot-active')
   output=torch.zeros_like(hidden)
   for idx,pos in e['coslot_combine']:output.index_add_(0,idx,returned[pos])
  else:
   send_w=dense[e['send_idx'][:,None],e['send_eids'].clamp_min(0)]*(e['send_eids']>=0)
   received=self.exchange(hidden[e['send_idx']],e['send_counts'],e['recv_counts'],activation_row_bytes=2048*hidden.element_size())
   rw=self.exchange(send_w,e['send_counts'],e['recv_counts']);parts=[]
   for expert,rows,cols,slot in e['groups']:
    assert self.keys[slot]==layer*128+expert
    if self.h2d:self.h2d.wait_for_slot(slot)
    w=self.cache[slot];gate=w[:1572864].view(768,2048);up=w[1572864:3145728].view(768,2048);down=w[3145728:].view(2048,768)
    part=self.kernel(received[rows],gate,up,down)
    if self.h2d:self.h2d.record_slot_use(slot)
    parts.append(part*rw[rows,cols,None])
   values=torch.cat(parts) if parts else hidden.new_empty((0,2048))
   returned=self.exchange(values[e['return_order']],e['return_counts'],e['return_recv_counts'],activation_row_bytes=2048*values.element_size())
   output=torch.zeros_like(hidden)
   for idx,pos in e['combine']:output.index_add_(0,idx,returned[pos])
  self.index+=1;return output

def generate(model,rt,initial_ids,initial_mask):
 check_cpu_residency(rt.cpu_backing)
 ids=initial_ids;mask=initial_mask;past=None;tokens=[]
 # Padding selection is precomputed before boundaries; decode has no padding.
 rt.valid=initial_mask.reshape(-1).nonzero().flatten()
 dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
 begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 with torch.inference_mode():
  for step in range(257):
   position=mask.cumsum(-1)-1;position.masked_fill_(mask==0,0)
   out=model(input_ids=ids,attention_mask=mask,position_ids=position[:,-ids.shape[1]:],past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
   if step<256:
    logits=out.logits[:,-1].clone();logits[:,model.generation_config.eos_token_id]=-torch.inf
    ids=logits.argmax(-1,keepdim=True);tokens.append(ids);mask=torch.cat((mask,mask.new_ones((mask.shape[0],1))),1)
   if step==0:torch.cuda.synchronize();decode_start=time.perf_counter();begin.record();rt.valid=None
   if rt.args.phase!='MEASURE' and step%16==0:print(json.dumps(dict(phase=rt.args.phase,rank=rt.rank,decode=step)),flush=True)
  end.record();torch.cuda.synchronize();finish=time.perf_counter()
 result=torch.cat(tokens,1).cpu().numpy()
 assert rt.index==257*48 and not rt.mismatch.item()
 return dict(token_hash=array_hash(result),state_hash=array_hash(rt.keys),E2E_wall=finish-start,decode_wall=finish-decode_start,TPOT=begin.elapsed_time(end)/1000/256),result

def main(a):
 if a.comm_mode in ('coslot','coslot-active') or a.h2d_mode=='pinned':
  rank=int(os.environ['RANK']);cpus=list(range(24*rank,24*(rank+1)))
  for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'));rank=dist.get_rank()
 cell=json.loads((ROOT/'frozen_matrix.json').read_text())['cells'][a.cell];a.dataset=cell['dataset'];a.batch=cell['local_batch'];a.substitution=cell['substitution']
 world=cell['ranks'];assert dist.get_world_size()==world
 total=int(48*128*cell['cache_ratio']);a.capacities=[total//world+(r<total%world) for r in range(world)]
 records=json.loads((SOURCE/(a.dataset+'_requests.json')).read_text())['requests'][:world*a.batch];records=records[rank::world]
 model,backing,experts=load_model();print(json.dumps(dict(event='MODEL_LOADED',phase=a.phase,rank=rank)),flush=True);rt=Runtime(a,model,experts,a.capacities[rank]);rt.cpu_backing=backing
 length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
 if a.phase in ('PLAN','COUNTERS'):
  receipt,tokens=generate(model,rt,ids,mask)
  if a.phase=='PLAN':
   with gzip.open(a.output/f'rank{rank}.pkl.gz','wb') as f:pickle.dump(rt.events,f,protocol=4)
  np.save(a.output/f'rank{rank}_metrics.npy',np.asarray(rt.metrics));np.save(a.output/f'rank{rank}_tokens.npy',tokens)
  if a.phase=='PLAN':receipt['action_hash']=digest(rt.events)
  else:
   expected=json.loads((a.plan/f'rank{rank}.json').read_text());assert all(receipt[k]==expected[k] for k in ['token_hash','state_hash'])
   receipt.update(actual_H2D_bytes=rt.actual_h2d_bytes,actual_peer_bytes=rt.actual_peer_bytes,actual_wire_bytes=rt.actual_wire_bytes,actual_collective_calls=rt.actual_collective_calls,actual_p2p_batches=rt.actual_p2p_batches,actual_p2p_ops=rt.actual_p2p_ops,actual_zero_remote_rounds=rt.actual_zero_remote_rounds,active_peer_degree_sum=rt.active_peer_degree_sum,max_active_peer_degree=rt.max_active_peer_degree,logical_counters=np.asarray(rt.metrics).sum(axis=0).tolist())
 else:
  # Populate/load compiler caches and exercise the complete frozen schedule.
  warm,tokens=generate(model,rt,ids,mask)
  expected=json.loads((a.plan/f'rank{rank}.json').read_text())
  assert all(warm[k]==expected[k] for k in ['token_hash','state_hash'])
  print(json.dumps(dict(event='WARMUP_VALIDATED',phase=a.phase,rank=rank)),flush=True)
  if a.phase=='COMPILE':receipt=warm
  else:
   from torch._dynamo.utils import counters
   before=dict(counters['stats']);rt.reset();torch.manual_seed(42)
   if a.comm_mode!='current' or a.h2d_mode=='pinned':
    write(a.output/f'ready_rank{rank}.json',dict(rank=rank,status='READY'));dist.barrier();deadline=time.monotonic()+600
    while not (a.output/'GO').exists():
     if time.monotonic()>deadline:raise TimeoutError('measurement boundary timeout')
     time.sleep(.1)
   with torch._dynamo.config.patch(error_on_recompile=True):receipt,tokens=generate(model,rt,ids,mask)
   assert all(receipt[k]==expected[k] for k in ['token_hash','state_hash'])
   assert dict(counters['stats'])==before,'compile occurred in MEASURE'
   receipt.update(no_compile_in_measure=True,compiler_stats=before)
 receipt.update(status='PASS',phase=a.phase,cell=a.cell,policy=a.policy,environment=a.environment,comm_mode=a.comm_mode,h2d_mode=a.h2d_mode,h2d_stages=a.h2d_stages,pinned_stage_bytes=(rt.h2d.pinned_bytes if rt.h2d else 0),rank=rank,boot=BOOT,cell_spec=cell,matrix_sha256=hashlib.sha256((ROOT/'frozen_matrix.json').read_bytes()).hexdigest(),tensor_layout_sha256=hashlib.sha256((PKG/'scripts/env_offload_tensors.py').read_bytes()).hexdigest(),cache_capacity=a.capacities[rank],max_resident_copies=int(np.count_nonzero(rt.keys>=0)),cpu_expert_pool_bytes=backing.numel(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
 if a.phase!='MEASURE':
  for k in ['E2E_wall','decode_wall','TPOT']:receipt.pop(k,None)
 if a.phase!='PLAN':receipt.update(route_hash=rt.plan_proof['route_hash'],route_validation='device equality for all 12336 events; digest from validated PLAN',action_hash=rt.plan_proof['action_hash'],schedule_file_sha256=rt.plan_proof['schedule_file_sha256'])
 write(a.output/f'rank{rank}.json',receipt);dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',choices=['P','R','E'],required=True);p.add_argument('--policy',choices=['BR','CA','CA-rep'],required=True);p.add_argument('--phase',choices=['PLAN','COMPILE','MEASURE','COUNTERS'],required=True);p.add_argument('--environment',choices=['env1','env2'],required=True);p.add_argument('--comm-mode',choices=['current','coslot','coslot-active'],default='current');p.add_argument('--h2d-mode',choices=['pageable','pinned'],default='pageable');p.add_argument('--h2d-stages',type=int,default=2);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path);main(p.parse_args())
