"""Stock ZeRO-3 BF16 CPU parameter offload; same native request manifests."""
import argparse,gc,json,os,time
from pathlib import Path
import torch,torch.distributed as dist,psutil
import deepspeed
from transformers import AutoModelForCausalLM
from transformers.integrations import HfDeepSpeedConfig
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
MODEL=os.environ.get('MGO_MODEL_PATH','/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507')
def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def release():
 dist.barrier();torch.cuda.synchronize()
 stamp=torch.tensor([time.perf_counter_ns()+200_000_000 if dist.get_rank()==0 else 0],device='cuda',dtype=torch.int64)
 dist.broadcast(stamp,0);start=stamp.item()
 while time.perf_counter_ns()<start:time.sleep(.0001)
 return start

def generate(engine,ids,n,progress=None):
 start=release();mask=torch.ones_like(ids);past=None;tokens=[];stamps=[];finite=torch.ones((),device='cuda',dtype=torch.bool)
 with torch.no_grad():
  for step in range(n):
   if progress is not None:progress(step)
   out=engine(input_ids=ids,attention_mask=mask,past_key_values=past,use_cache=True,logits_to_keep=1)
   past=out.past_key_values;next_ids=out.logits[:,-1].argmax(-1);tokens.append(next_ids);finite.logical_and_(torch.isfinite(out.logits).all())
   torch.cuda.synchronize();stamps.append(time.perf_counter_ns())
   ids=next_ids[:,None];mask=torch.cat((mask,mask.new_ones((len(ids),1))),1)
 assert finite.item()
 assert all(layer.keys.is_cuda and layer.values.is_cuda for layer in past.layers)
 return dict(release_ns=start,first_ns=stamps[0],end_ns=stamps[-1],token_ready_ns=stamps,tokens=torch.stack(tokens,1).cpu().tolist(),finite_logits=True)

def main(a):
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,4,5'
 physical=[0,1,4,5]
 if os.environ.get('MGO_PCIE_HOST')=='1':
  from pcie_host import affinity
  cpus=affinity(local)
 else:cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[local])]
 # Match OURS' disjoint per-rank CPU ranges; the PCIe host has two NUMA nodes.
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),cpus)
  except FileNotFoundError:pass
 torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(42)
 write(a.output/f'affinity_rank{rank}.json',dict(physical_gpu=physical[local],cpus=cpus,mode='fixed per-rank physical CPU cores matching OURS' if os.environ.get('MGO_PCIE_HOST')=='1' else 'legacy fixed affinity'))
 deepspeed.init_distributed();assert dist.get_world_size()==4
 spec=next(x for x in json.loads(Path(os.environ.get('MGO_HEADLINE_WORKLOADS',str(ROOT/'WORKLOADS.json'))).read_text())['cells'] if x['cell']==a.cell)
 model_path=spec.get('model_path',MODEL)
 assert model_path in (MODEL,'/home/hwlee/model/DeepSeek-V2-Lite-Chat')
 model_family=spec.get('model','Qwen3')
 assert (model_family=='Qwen3')==(model_path==MODEL)
 batch=1 if a.smoke else spec['local_batch'];n=2 if a.smoke else 64
 scale=spec.get('expert_budget_bytes',17392730112)/17392730112
 cfg=dict(train_batch_size=batch*4,train_micro_batch_size_per_gpu=batch,gradient_accumulation_steps=1,bf16={'enabled':True},zero_optimization=dict(stage=3,offload_param={'device':'cpu','pin_memory':True},stage3_max_live_parameters=int(1_500_000_000*scale),stage3_prefetch_bucket_size=int(1_000_000_000*scale),stage3_max_reuse_distance=1_000_000_000,stage3_param_persistence_threshold=100_000),steps_per_print=1000000,wall_clock_breakdown=False)
 dschf=HfDeepSpeedConfig(cfg)
 model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
 if model_family!='Qwen3':
  from deepseek_moe_loop import install_compact_deepseek_moe
  assert install_compact_deepseek_moe(model)==26
 if os.environ.get('MGO_PCIE_HOST')=='1' and model_family=='Qwen3':
  # Different local routes call different expert modules on different ranks.
  # ZeRO must gather the whole MoE block before those conditional calls.
  from pcie_deepspeed_leaf import mark_qwen3_moe_leaves
  leaf_receipt=mark_qwen3_moe_leaves(model,spec.get('expert_budget_bytes',17392730112)//4)
  write(a.output/f'zero_leaf_rank{rank}.json',leaf_receipt)
 if rank==0:write(a.output/'phase.json',dict(system='DeepSpeed-ZeRO-Inference',phase='engine_initialization',cell=a.cell))
 model.eval();engine,_,_,_=deepspeed.initialize(model=model,config=cfg);engine.eval()
 if rank==0:write(a.output/'phase.json',dict(system='DeepSpeed-ZeRO-Inference',phase='budget_calibration',cell=a.cell))
 if a.smoke and model_family!='Qwen3':
  # Smoke-only layer checkpoints separate slow forward progress from a wait.
  for layer_index,layer in enumerate(engine.module.model.layers):
   def mark_layer(module,args,layer_index=layer_index):
    write(a.output/f'layer_progress_rank{rank}.json',dict(layer=layer_index,unix=time.time()))
   layer.register_forward_pre_hook(mark_layer)
 params=list(engine.module.named_parameters());assert all(hasattr(p,'ds_id') for _,p in params)
 assert all(p.dtype==torch.bfloat16 for _,p in params)
 pinned=sum(p.ds_tensor.numel()*p.ds_tensor.element_size() for _,p in params if p.ds_tensor.device.type=='cpu' and p.ds_tensor.is_pinned())
 coordinator=engine.optimizer.get_param_coordinator()
 budget=spec.get('expert_budget_bytes',17392730112)//4;peak=[0];all_peak=[0];samples=[0];full_scans=[0]
 records=[(('.experts.' in name),p,p.ds_numel*p.element_size()) for name,p in params]
 def scan_residency():
  charged=all_charged=0
  for expert,param,nbytes in records:
   if param.ds_status in (ZeroParamStatus.AVAILABLE,ZeroParamStatus.INFLIGHT):
    all_charged+=nbytes
    if expert:charged+=nbytes
  return charged,all_charged
 def measure_residency():
  samples[0]+=1
  native_charged=getattr(coordinator,'_PartitionedParameterCoordinator__n_available_params')*2
  assert 0<=native_charged<=budget,(native_charged,budget)
  all_peak[0]=max(all_peak[0],native_charged)
  # The native counter is O(1) at every fetch. DeepSeek has thousands of
  # expert parameters; a complete Python scan at every fetch adds substantial
  # host work to the two-token calibration. Cross-check periodically.
  if model_family=='Qwen3' or samples[0]==1 or samples[0]%64==0:
   charged,all_charged=scan_residency()
   assert all_charged==native_charged,(all_charged,native_charged)
   peak[0]=max(peak[0],charged);full_scans[0]+=1
 # Enforce the exact all-parameter C30 bound at every native fetch; the
 # separately reported DeepSeek expert peak is a periodic sample.
 original_fetch=coordinator.fetch_sub_module
 def calibrated_fetch(*args,**kwargs):
  result=original_fetch(*args,**kwargs);measure_residency();return result
 coordinator.fetch_sub_module=calibrated_fetch
 rows=json.loads(Path(spec['warmup']['path']).read_text())['requests']
 local_rows=rows[rank*batch:(rank+1)*batch];ids=torch.tensor([r['input_ids'][-32:] for r in local_rows],device='cuda')
 if rank==0:write(a.output/'phase.json',dict(system='DeepSpeed-ZeRO-Inference',phase='calibration_generate',cell=a.cell,smoke=a.smoke))
 calibration_start=time.perf_counter()
 generate(engine,ids,2,progress=(lambda step:write(a.output/'phase.json',dict(system='DeepSpeed-ZeRO-Inference',phase='calibration_generate',step=step,cell=a.cell,smoke=a.smoke)) if rank==0 else None))
 calibration_seconds=time.perf_counter()-calibration_start
 coordinator.fetch_sub_module=original_fetch
 assert 0<all_peak[0]<=budget and 0<=peak[0]<=all_peak[0],(peak,all_peak,budget)
 write(a.output/f'calibration_rank{rank}.json',dict(status='PASS',model_path=model_path,expert_peak_bytes=peak[0],expert_peak_sampled=model_family!='Qwen3',all_parameter_peak_bytes=all_peak[0],budget_bytes=budget,samples=samples[0],full_scans=full_scans[0],calibration_seconds=calibration_seconds,pinned_host_bytes=pinned,config=cfg))
 # Validate the native live-parameter counter against the complete scan above,
 # then sample that O(1) counter at fetch boundaries in primary execution.
 live_peak=[0];live_samples=[0];counter_offset=[0];counter_corrections=[0];counter_full_scans=[0]
 def bounded_fetch(*args,**kwargs):
  result=original_fetch(*args,**kwargs)
  live_samples[0]+=1
  charged=getattr(coordinator,'_PartitionedParameterCoordinator__n_available_params')*2+counter_offset[0]
  if live_samples[0]==1 or live_samples[0]%256==0 or charged<0 or charged>budget:
   _,actual=scan_residency()
   counter_full_scans[0]+=1
   if actual!=charged:
    counter_offset[0]+=actual-charged
    counter_corrections[0]+=1
    charged=actual
  assert 0<=charged<=budget,(charged,budget)
  live_peak[0]=max(live_peak[0],charged)
  return result
 coordinator.fetch_sub_module=bounded_fetch
 for repeat in range((1 if a.smoke else a.repeats)+1):
  phase='warmup' if repeat==0 else 'target';rows=json.loads(Path(spec[phase]['path']).read_text())['requests'];local_rows=rows[rank*batch:(rank+1)*batch]
  ids=torch.tensor([r['input_ids'][-32:] if a.smoke else r['input_ids'] for r in local_rows],device='cuda')
  coordinator.release_and_reset_all(engine.module)
  assert all(p.ds_status==ZeroParamStatus.NOT_AVAILABLE for _,p in params)
  counter_offset[0]=-getattr(coordinator,'_PartitionedParameterCoordinator__n_available_params')*2
  live_samples[0]=0;counter_corrections[0]=0;counter_full_scans[0]=0
  gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  if rank==0:write(a.output/'phase.json',dict(system='DeepSpeed-ZeRO-Inference',phase=phase,repeat=repeat,cell=a.cell,smoke=a.smoke))
  live_peak[0]=0
  row=generate(engine,ids,n);row.update(rank=rank,repeat=repeat,phase=phase,smoke=a.smoke,all_parameter_peak_bytes=live_peak[0],parameter_budget_bytes=budget,parameter_counter_offset_bytes=counter_offset[0],parameter_counter_corrections=counter_corrections[0],parameter_counter_full_scans=counter_full_scans[0],kv_gpu_resident=True,cpu_affinity=sorted(os.sched_getaffinity(0)),cache_start='all parameters NOT_AVAILABLE',pinned_host_bytes=pinned,host_rss_bytes=psutil.Process().memory_info().rss,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),request_ids=[r['request_id'] for r in local_rows])
  write(a.output/f'repeat{repeat}_rank{rank}.json',row);dist.barrier()
  if rank==0:
   rr=[json.loads((a.output/f'repeat{repeat}_rank{r}.json').read_text()) for r in range(4)];assert len(set(r['release_ns'] for r in rr))==1
   start=rr[0]['release_ns'];first=max(r['first_ns'] for r in rr);end=max(r['end_ns'] for r in rr)
   result=dict(status='PASS',system='DeepSpeed-ZeRO-Inference',repeat=repeat,smoke=a.smoke,TTFT=(first-start)/1e9,TPOT=(end-first)/1e9/(n-1),E2E=(end-start)/1e9,throughput=4*batch*n/((end-start)/1e9),global_requests=4*batch,output_tokens=n)
   write(a.output/f'repeat{repeat}.json',result);print(json.dumps(result),flush=True)
  dist.barrier()
 if rank==0:write(a.output/'result.json',dict(status='PASS',system='DeepSpeed-ZeRO-Inference',cell=a.cell,smoke=a.smoke))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--repeats',type=int,choices=range(1,6),default=3);main(p.parse_args())
