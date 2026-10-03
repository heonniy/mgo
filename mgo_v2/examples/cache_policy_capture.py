import os,json,hashlib,types,argparse
from pathlib import Path
import benchmark_model as b
os.environ.pop('NCCL_P2P_DISABLE',None)
import mgo_v2.communicator as comm
from mgo_v2.controller_diagnostics import digest
from dataclasses import asdict

ROOT=None
INPUTS=Path('/home/hwlee/mgo-results/runtime_validation_20261001')
MODEL='/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
def main(a):
 global ROOT
 ROOT=a.output
 assert b.BOOT['visible_gpu']==str([0,1,4,5][b.BOOT['local_rank']])
 assert {k:v for k,v in os.environ.items() if k.startswith('NCCL_')}=={'NCCL_CUMEM_ENABLE':'0'}
 b.torch.set_num_threads(4);b.torch.cuda.set_device(0)
 b.dist.init_process_group('nccl',device_id=b.torch.device('cuda:0'))
 rank=b.dist.get_rank();assert b.dist.get_world_size()==4;b.warmup_collectives()
 model,handle,dispatcher,store=b.load_qwen3_slots(MODEL,str(INPUTS/'expert_store'))
 cfg=model.config;assert cfg.hidden_size==2048 and cfg.num_experts_per_tok==8
 config=b.RuntimeConfig(num_layers=48,num_experts=128,top_k=8,world_size=4,global_cache_ratio=.3,admission='random',eviction='lru',substitution_enabled=False,seed=42)
 c=b.GlobalExpertController(config,b.np.load(INPUTS/'similarity.npy'),None)
 executor=b.LegacySlotExecutorAdapter(dispatcher,config.per_rank_slots()[rank],0,128)
 records=[];old_plan=c.plan_layer
 def plan(self,routes):
  p=old_plan(routes)
  raw=set(map(int,routes.selected_experts.ravel()));assert raw==set(p.owner_by_expert)
  assert not p.substitution.source_to_target and all(set(map(int,s))==set(e) for s,e in zip(routes.selected_experts,p.effective_token_routes))
  expected_dispatch=[0]*4;expected_return=[0]*4
  for origin,token in zip(routes.origin_ranks,p.effective_token_routes):
   dest=[p.owner_by_expert[e] for e in token]
   if int(origin)==rank:
    for dst in set(dest):expected_dispatch[dst]+=1
   expected_return[int(origin)]+=dest.count(rank)
  state=(c.cache.tick,dict(c.cache.owner),[[tuple(x) if x is not None else None for x in r.slots] for r in c.cache.ranks],[[(*key,v.slot,v.last_used,v.admitted_at) for key,v in sorted(r.entries.items())] for r in c.cache.ranks])
  records.append(dict(event=len(records),step=len(records)//48,layer=routes.layer,raw_selected_experts=routes.selected_experts.tolist(),origin_ranks=routes.origin_ranks.tolist(),routing_weights=routes.routing_weights.tolist(),gate_scores=[c.history.score(routes.layer,e) for e in range(128)],owner_by_expert=p.owner_by_expert,effective_token_routes=p.effective_token_routes,plan_sha256=digest(p),cache_sha256=digest(state),expected_dispatch=expected_dispatch,expected_return=expected_return))
  return p
 c.plan_layer=types.MethodType(plan,c)
 original=comm._all_to_all_varlen
 def exchange(send,send_counts,recv_counts,stats=None,kind='payload'):
  if kind in ('dispatch_hidden','return_outputs'):
   r=records[-1];phase='dispatch' if kind=='dispatch_hidden' else 'return'
   assert phase+'_send_counts' not in r
   assert list(send_counts)==r['expected_'+phase]
   assert send.dtype==b.torch.bfloat16 and send.shape[1]==2048 and send.shape[0]==sum(send_counts)
   r[phase+'_send_counts']=list(send_counts);r[phase+'_recv_counts']=list(recv_counts)
   r[phase+'_row_bytes']=send.element_size()*send.shape[1]
  return original(send,send_counts,recv_counts,stats,kind)
 comm._all_to_all_varlen=exchange
 runtime=b.DistributedMoERuntime(c,executor,debug=True)
 assert b.attach_qwen3_runtime(model,runtime)==48
 tok=b.AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left')
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 manifest=json.loads(a.prompts.read_text()); prompt_rows=manifest['batches'][str(a.batch)]['ranks'][rank]
 workload=json.loads((INPUTS/'screen_workload.json').read_text())
 assert hashlib.sha256((INPUTS/'screen_workload.json').read_bytes()).hexdigest()==manifest['workload_sha256']
 rows=[workload[r['workload_index']] for r in prompt_rows]
 assert all(hashlib.sha256(r['question'].encode()).hexdigest()==m['question_sha256'] for r,m in zip(rows,prompt_rows))
 texts=[tok.apply_chat_template([{'role':'user','content':r['question']+'\nReturn only the final numeric answer, without explanation.'}],tokenize=False,add_generation_prompt=True) for r in rows]
 encoded={k:v.cuda() for k,v in tok(texts,padding=True,return_tensors='pt').items()}
 # Diagnostic-only: no model warmup or performance/quality evaluation.
 b.dist.barrier()
 with b.torch.inference_mode():output,_=b.generate(model,encoded['input_ids'],encoded['attention_mask'],9)
 b.torch.cuda.synchronize();executor.assert_cache_matches(c.cache.ranks[rank]);c.cache.assert_consistent()
 assert len(records)==432 and all(r['layer']==r['event']%48 for r in records)
 for r in records:
  assert 'dispatch_send_counts' in r and 'return_send_counts' in r
  if r['step']>0:assert len(r['origin_ranks'])==4*a.batch and [r['origin_ranks'].count(i) for i in range(4)]==[a.batch]*4
 hashes=[(r['plan_sha256'],r['cache_sha256'],digest(r['gate_scores'])) for r in records];all_hashes=[None]*4;b.dist.all_gather_object(all_hashes,hashes);assert all(h==hashes for h in all_hashes)
 assert output.shape==(a.batch,9) and bool(((output>=0)&(output<cfg.vocab_size)).all())
 data=dict(status='PASS',rank=rank,local_batch=a.batch,prompts=prompt_rows,prompt_manifest_sha256=hashlib.sha256(a.prompts.read_bytes()).hexdigest(),generated_token_sha256=digest(output.cpu().tolist()),transport_env={'NCCL_CUMEM_ENABLE':'0'},boot=b.BOOT,config=asdict(config),hidden_size=2048,element_bytes=2,prefill_forwards=1,decode_forwards=8,model_warmup_forwards=0,generated_tokens=output.cpu().tolist(),events=records,cache_stats=dispatcher.get_cache_stats().tolist(),expected_fetches=executor.expected_fetches,plan_cache_equal_all_ranks=True,checkpoint=store['identity'])
 reference=json.loads((a.reference/f'rank{rank}.json').read_text())
 assert data['generated_tokens']==reference['generated_tokens'],'Generated-token mismatch: hard stop'
 assert len(reference['events'])==len(records)
 for actual,expected in zip(records,reference['events']):
  for field in ('event','step','layer','raw_selected_experts','origin_ranks','routing_weights','owner_by_expert','plan_sha256','cache_sha256'):
   # JSON canonicalizes integer dictionary keys in prior receipts.
   assert json.loads(json.dumps(actual[field]))==expected[field],f'Reference mismatch event={actual["event"]} field={field}'
  assert len(actual['gate_scores'])==128 and all(b.np.isfinite(actual['gate_scores']))
 data['reference_parity']=dict(tokens=True,routes=True,weights=True,owners=True,plan=True,cache=True,gate_equal_all_ranks=True,path=str(a.reference/f'rank{rank}.json'),sha256=hashlib.sha256((a.reference/f'rank{rank}.json').read_bytes()).hexdigest())
 (ROOT/f'rank{rank}.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
 print(json.dumps(dict(status='PASS',rank=rank,events=len(records),decode_events=384,exact_owner_complete=True)),flush=True)
 b.dist.barrier();b.dist.destroy_process_group();b.detach_qwen3_runtime(model)
 del model,runtime,executor,dispatcher;handle.clean_up_resources()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prompts',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--batch',type=int,choices=[8,32],required=True);main(p.parse_args())
