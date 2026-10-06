"""Owner-repaired BF16 MoE-Infinity, native requests and cold expert cache."""
import argparse,copy,gc,hashlib,json,os,time,weakref
from pathlib import Path
import torch,psutil
from moe_infinity import MoE
from transformers import LogitsProcessorList
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
MODEL='/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def sync():
 for gpu in range(4):torch.cuda.synchronize(gpu)
class FiniteLogits:
 def __init__(self):self.flag=None
 def __call__(self,input_ids,scores):
  finite=torch.isfinite(scores).all()
  if self.flag is None:self.flag=finite
  else:self.flag.logical_and_(finite)
  return scores
class ClockStreamer:
 def __init__(self):self.stamps=[];self.prompt=True
 def put(self,value):
  if self.prompt:self.prompt=False;return
  sync();self.stamps.append(time.perf_counter_ns())
 def end(self):pass

def main(a):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,4,5' and torch.cuda.device_count()==4
 torch.set_num_threads(8);torch.manual_seed(42)
 spec=next(x for x in json.loads((ROOT/'WORKLOADS.json').read_text())['cells'] if x['cell']==a.cell)
 cfg=dict(offload_path='/home/hwlee/mgo-tools/headline-r4/infinity-bf16-store',device_memory_ratio=.25,host_memory_ratio=.2,prefetch=True,use_native_engine=False,enable_attention_offload=False,enable_kv_cache_offload=False,speculative_prefetch=False,gpu_only_expert_routing=True,num_threads=4)
 write(a.output/'config.json',cfg)
 sources={}
 roots=[Path('/home/hwlee/mgo-tools/headline-r4/MoE-Infinity/moe_infinity'),Path('/home/hwlee/mgo-tools/headline-r4/infinity-env/lib/python3.12/site-packages/moe_store/wrappers')]
 for root in roots:
  for file in root.rglob('*'):
   if file.suffix in ('.py','.so'):sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
 write(a.output/'source_hashes.json',sources)
 model=MoE(MODEL,cfg);engine=model.engine;p=engine.expert_prefetcher
 assert engine.dtype==0,engine.dtype  # native BF16 enum
 assert set(p.expert_nbytes_map.values())=={9*2**20},set(p.expert_nbytes_map.values())
 budgets=[x*9*2**20 for x in [461,461,461,460]]
 p.configure_eam_budget(budgets,9*2**20)
 initial=p.reset_eam_residency();assert initial['capacity_bytes']==sum(budgets)
 write(a.output/'initial_budget.json',dict(budgets=budgets,stats=initial))
 attention_checks=[0]
 def attention_gpu(module,args,kwargs):
  hidden=args[0] if args else kwargs['hidden_states']
  assert hidden.is_cuda,'CPU attention is forbidden'
  attention_checks[0]+=1
 attention_hooks=[m.register_forward_pre_hook(attention_gpu,with_kwargs=True) for name,m in model.model.named_modules() if name.endswith('.self_attn')]
 assert len(attention_hooks)==48
 n=2 if a.smoke else 64;repeats=1 if a.smoke else 3;saved=None
 for repeat in range(repeats+1):
  phase='warmup' if repeat==0 else 'target'
  rows=json.loads(Path(spec[phase]['path']).read_text())['requests']
  if a.smoke:rows=rows[:4]
  ids=torch.tensor([r['input_ids'][-32:] if a.smoke else r['input_ids'] for r in rows],device='cuda:0')
  tracer=engine.expert_tracer
  if saved is not None:
   tracer.trace_collection[:]=saved[0];tracer.collection_access[:]=saved[1];tracer.access_clock=saved[2]
  assert not tracer.trace
  before=p.reset_eam_residency();sync()
  for gpu in range(4):torch.cuda.reset_peak_memory_stats(gpu)
  write(a.output/'phase.json',dict(system='MoE-Infinity-repaired',phase=phase,repeat=repeat,cell=a.cell,smoke=a.smoke))
  streamer=ClockStreamer();finite=FiniteLogits();start=time.perf_counter_ns()
  with torch.no_grad():
   output=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=n,min_new_tokens=n,do_sample=False,eos_token_id=None,pad_token_id=0,streamer=streamer,logits_to_keep=1,logits_processor=LogitsProcessorList([finite]),return_dict_in_generate=True)
  kv=output.past_key_values
  assert all(layer.keys.is_cuda and layer.values.is_cuda for layer in kv.layers)
  output=output.sequences
  sync();assert finite.flag is not None and finite.flag.item()
  assert len(streamer.stamps)==n and output.shape==(len(rows),ids.shape[1]+n)
  assert torch.equal(output[:,:ids.shape[1]],ids)
  tokens=output[:,ids.shape[1]:].cpu().tolist()
  # Do not retain the preceding batch's KV while generating the next batch.
  kv_ref=weakref.ref(kv);kv_tensors=[weakref.ref(t) for layer in kv.layers for t in (layer.keys,layer.values)]
  del output,kv
  gc.collect();sync()
  assert kv_ref() is None and all(ref() is None for ref in kv_tensors),'previous batch KV retained'
  live_after_kv_release=[torch.cuda.memory_allocated(g) for g in range(4)]
  assert not tracer.trace
  stats=dict(p.archer_engine.get_expert_policy_stats());assert p.eam_calls==48*n
  assert stats['capacity_bytes']==sum(budgets)
  for gpu,budget in enumerate(budgets):
   assert 0 <= stats[f'gpu_{gpu}_peak_charged_bytes'] <= budget
  assert stats['peak_accounted_bytes'] <= sum(budgets)
  assert stats['resident_bytes']+stats['transition_reserved_bytes']+stats['workspace_bytes']<=sum(budgets)
  first,end=streamer.stamps[0],streamer.stamps[-1]
  result=dict(status='PASS',system='MoE-Infinity-repaired',cell=a.cell,repeat=repeat,phase=phase,smoke=a.smoke,TTFT=(first-start)/1e9,TPOT=(end-first)/1e9/(n-1),E2E=(end-start)/1e9,throughput=len(rows)*n/((end-start)/1e9),global_requests=len(rows),output_tokens=n,request_ids=[r['request_id'] for r in rows],tokens=tokens,token_ready_ns=streamer.stamps,release_ns=start,cache_before=before,cache_after=stats,expert_budget_per_gpu=budgets,eam_calls=p.eam_calls,eam_candidates=p.eam_candidates,kv_released=True,allocated_after_kv_release=live_after_kv_release,peak_allocated_bytes=[torch.cuda.max_memory_allocated(g) for g in range(4)],peak_reserved_bytes=[torch.cuda.max_memory_reserved(g) for g in range(4)],host_rss_bytes=psutil.Process().memory_info().rss)
  write(a.output/f'repeat{repeat}.json',result)
  print(json.dumps({k:result[k] for k in ['repeat','TTFT','TPOT','E2E','eam_calls','eam_candidates']}),flush=True)
  if repeat==0:
   assert attention_checks[0]==48*n
   for hook in attention_hooks:hook.remove()
   saved=(tracer.trace_collection.copy(),tracer.collection_access.copy(),tracer.access_clock)
 write(a.output/'result.json',dict(status='PASS',system='MoE-Infinity-repaired',cell=a.cell,smoke=a.smoke,primary_repeats=repeats))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
