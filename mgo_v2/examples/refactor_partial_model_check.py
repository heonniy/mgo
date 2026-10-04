"""Frozen-route full-model numerical/argmax report for coalesced return modes."""
from refactor_measure_worker import *

def capture(model,rt,ids0,mask0,teacher,horizon,reference=None):
 ids=ids0;mask=mask0;past=None;rt.valid=mask0.reshape(-1).nonzero().flatten();saved=[];tokens=[];rows=[]
 with torch.inference_mode():
  for step in range(horizon+1):
   position=mask.cumsum(-1)-1;position.masked_fill_(mask==0,0)
   out=model(input_ids=ids,attention_mask=mask,position_ids=position[:,-ids.shape[1]:],past_key_values=past,use_cache=True,logits_to_keep=1);past=out.past_key_values
   raw=out.logits[:,-1];assert bool(torch.isfinite(raw).all());cpu=raw.float().cpu()
   logits=raw.clone();logits[:,model.generation_config.eos_token_id]=-torch.inf;token=logits.argmax(-1).cpu();tokens.append(token)
   if reference is None:saved.append(cpu)
   else:
    ref=reference['logits'][step];delta=cpu-ref
    rows.append(dict(step=step,max_abs=float(delta.abs().max()),relative_L2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref).clamp_min(1e-20)),argmax_equal=int((token==reference['tokens'][step]).sum()),tokens=len(token)))
   if step<horizon:ids=teacher[:,step:step+1];mask=torch.cat((mask,mask.new_ones((mask.shape[0],1))),1)
   if step==0:rt.valid=None
 torch.cuda.synchronize();return dict(logits=saved,tokens=torch.stack(tokens),rows=rows)

def main(a):
 rank=int(os.environ['RANK']);a.output.mkdir(parents=True,exist_ok=True);horizon=8
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(rank)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=[3686//8+(r<3686%8) for r in range(8)];a.comm_mode='current';a.phase='COUNTERS';a.debug_plan=False;a.physical_prefetch=True;a.fused=True;a.streaming=True;a.ready_first=True;a.trigger='T1'
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records);model,backing,experts=load_model();length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda');teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda')
 a.policy='BR';a.arena_budget=0;rt=PrefixRuntime(a,model,backing,experts);reference=capture(model,rt,ids,mask,teacher,horizon);proof=json.loads((a.inputs/'BR_P0_proof.json').read_text());validate(rt,{},proof,rank)
 for block in model.model.layers:block.mlp.forward=lambda *args:None
 del rt;gc.collect();torch.cuda.empty_cache();reports=[];mode_tokens={}
 for precision in ['bf16','fp32','fp64']:
  for policy in ['BR','LA']:
   a.policy=policy;a.arena_budget=2;a.partial_precision=precision;rt=DecodeOffloadRuntime(a,model,backing,experts);rt.stage_frozen_inputs(horizon)
   result=capture(model,rt,ids,mask,teacher,horizon,reference);proof=json.loads((a.inputs/f'{policy}_P2_proof.json').read_text());validate(rt,{},proof,rank)
   # Cross-policy token differences are distinct from legacy-reference differences.
   cross=None
   if policy=='BR':mode_tokens[precision]=result['tokens']
   else:cross=float((mode_tokens[precision]==result['tokens']).float().mean())
   report=dict(policy=policy,precision=precision,steps=result['rows'],argmax_agreement=sum(x['argmax_equal'] for x in result['rows'])/sum(x['tokens'] for x in result['rows']),max_relative_L2=max(x['relative_L2'] for x in result['rows']),max_abs=max(x['max_abs'] for x in result['rows']),BR_LA_argmax_agreement=cross,payload_collectives=rt.transport.calls,forward_bytes=rt.transport.forward_bytes,return_bytes=rt.transport.return_bytes,copy_accounting='PASS',logical_state='PASS')
   reports.append(report);write(a.output/f'progress_rank{rank}.json',dict(status='RUNNING',rank=rank,completed=reports));rt.close()
   for block in model.model.layers:block.mlp.forward=lambda *args:None
   del rt;gc.collect();torch.cuda.empty_cache();dist.barrier()
 write(a.output/f'rank{rank}.json',dict(status='PASS',rank=rank,batch=batch,reports=reports,scope='frozen-route teacher-forced numerical differences; no free-generation quality claim',peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()));dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
