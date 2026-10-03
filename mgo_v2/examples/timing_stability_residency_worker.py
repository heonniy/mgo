"""Frozen physical diagnostic; baseline kernels/cache, coordinated monitor boundary."""
from env_offload_worker import *
import resource
import env_offload_worker as base
STABILITY=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004')
def generate(model,rt,initial_ids,initial_mask):
 recording=getattr(rt,'record_faults',False)
 if recording and rt.args.residency=='PRETOUCH':
  # CPU reads only; no GPU-cache state or expert tensors are modified.
  rt.pretouch_checksum=int(rt.cpu_backing[rt.touch_offsets].to(torch.int64).sum().item())
 check_cpu_residency(rt.cpu_backing)
 ids=initial_ids;mask=initial_mask;past=None;tokens=[]
 # Padding selection is precomputed before boundaries; decode has no padding.
 rt.valid=initial_mask.reshape(-1).nonzero().flatten()
 dist.barrier();torch.cuda.synchronize()
 if recording:
  usage_before=resource.getrusage(resource.RUSAGE_SELF)
  mem_before={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemAvailable','Cached']}
 dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
 begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 with torch.inference_mode():
  for step in range(rt.args.horizon+1):
   position=mask.cumsum(-1)-1;position.masked_fill_(mask==0,0)
   out=model(input_ids=ids,attention_mask=mask,position_ids=position[:,-ids.shape[1]:],past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
   if step<rt.args.horizon:
    logits=out.logits[:,-1].clone();logits[:,model.generation_config.eos_token_id]=-torch.inf
    ids=logits.argmax(-1,keepdim=True);tokens.append(ids);mask=torch.cat((mask,mask.new_ones((mask.shape[0],1))),1)
   if step==0:torch.cuda.synchronize();decode_start=time.perf_counter();begin.record();rt.valid=None
   if rt.args.phase!='MEASURE' and step%16==0:print(json.dumps(dict(phase=rt.args.phase,rank=rt.rank,decode=step)),flush=True)
  end.record();torch.cuda.synchronize();finish=time.perf_counter()
 dist.barrier()
 if recording:
  usage_after=resource.getrusage(resource.RUSAGE_SELF)
  mem_after={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemAvailable','Cached']}
  rt.fault_receipt=dict(minor_before=usage_before.ru_minflt,minor_after=usage_after.ru_minflt,major_before=usage_before.ru_majflt,major_after=usage_after.ru_majflt,minor_delta=usage_after.ru_minflt-usage_before.ru_minflt,major_delta=usage_after.ru_majflt-usage_before.ru_majflt,host_before=mem_before,host_after=mem_after,scope='rank process; includes boundary barriers, excludes warmup and PRETOUCH reads')
 result=torch.cat(tokens,1).cpu().numpy()
 assert rt.index==(rt.args.horizon+1)*48 and not rt.mismatch.item()
 return dict(token_hash=array_hash(result),state_hash=array_hash(rt.keys),E2E_wall=finish-start,decode_wall=finish-decode_start,TPOT=begin.elapsed_time(end)/1000/rt.args.horizon),result

def main(a):
 rank=int(os.environ['RANK'])
 topology=json.loads((STABILITY/'topology.json').read_text())
 if a.affinity=='fixed':
  cpus=topology['fixed_affinity'][str(rank)]
  for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 affinity=sorted(os.sched_getaffinity(0))
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 assert dist.get_world_size()==8
 a.phase='MEASURE';a.cell='P';a.policy='CA-rep';a.dataset='ShareGPT';a.batch=16;a.substitution=True
 total=int(48*128*.4);a.capacities=[total//8+(r<total%8) for r in range(8)]
 records=json.loads((SOURCE/'ShareGPT_requests.json').read_text())['requests'][:128][rank::8]
 model,backing,experts=load_model()
 # Reuse original kernel function and compile artifacts; only truncate immutable indices.
 original_pack=base.pack_layouts
 base.pack_layouts=lambda events: original_pack(events[:(a.horizon+1)*48])
 rt=Runtime(a,model,experts,a.capacities[rank]);rt.cpu_backing=backing
 keys=sorted({key for e in rt.events for key,slot,victim,rep in e['fetches']})
 page=4096;offsets=[]
 for key in keys:
  for tensor in experts[key]:
   offset=tensor.data_ptr()-backing.data_ptr();assert offset%page==0 and tensor.numel()*tensor.element_size()%page==0
   offsets.append(np.arange(offset,offset+tensor.numel()*tensor.element_size(),page,dtype=np.int64))
 rt.touch_offsets=torch.from_numpy(np.unique(np.concatenate(offsets)));rt.touch_expert_count=len(keys)
 length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
 expected=json.loads((STABILITY/'prefix_validation.json').read_text())['ranks'][rank] if a.horizon==64 else json.loads((a.plan/f'rank{rank}.json').read_text())
 warm,tokens=generate(model,rt,ids,mask)
 assert all(warm[k]==expected[k] for k in ['token_hash','state_hash'])
 from torch._dynamo.utils import counters
 before=dict(counters['stats']);rt.reset();torch.manual_seed(42);torch.cuda.synchronize();dist.barrier()
 write(a.output/f'ready{rank}.json',dict(rank=rank,affinity=affinity,boot=BOOT,warmup_hashes_valid=True,unix=time.time()))
 deadline=time.monotonic()+600
 while not (a.output/'GO').exists():
  if time.monotonic()>deadline:raise TimeoutError('boundary coordinator did not release workers')
  time.sleep(.1)
 rt.record_faults=True
 with torch._dynamo.config.patch(error_on_recompile=True):receipt,tokens=generate(model,rt,ids,mask)
 assert all(receipt[k]==expected[k] for k in ['token_hash','state_hash'])
 assert dict(counters['stats'])==before,'compilation in timed region'
 receipt.update(residency=a.residency,page_faults=rt.fault_receipt,pretouch_pages=rt.touch_offsets.numel() if a.residency=='PRETOUCH' else 0,scheduled_CPU_experts=rt.touch_expert_count,status='PASS',rank=rank,phase='MEASURE',horizon=a.horizon,mode=a.mode,affinity=affinity,affinity_mode=a.affinity,environment=a.environment,boot=BOOT,no_compile_in_measure=True,compiler_stats=before,route_validation='all frozen raw expert selections equal on device',schedule_file_sha256=rt.plan_proof['schedule_file_sha256'],peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),cpu_pool='existing pageable file-backed pool; node0 strict policy; no new locality claim')
 write(a.output/f'rank{rank}.json',receipt);dist.barrier();torch.cuda.synchronize();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--residency',choices=['NORMAL','PRETOUCH'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--horizon',type=int,choices=[64,256],required=True);p.add_argument('--mode',choices=['HEAVY','BOUNDARY'],required=True);p.add_argument('--affinity',choices=['original','fixed'],required=True);p.add_argument('--environment',choices=['env1','env2'],required=True);main(p.parse_args())
