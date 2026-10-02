"""CPU validation, frozen count extraction, and batch communication geometry."""
import csv,hashlib,json,statistics
from pathlib import Path
from batch_comm_common import PACKET,ROOT,BATCHES,write,sha
from trace_comm_input import validate_events
from summarize_trace_comm_replay import percentile

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def csvwrite(name,rows):
 with (PACKET/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def summary(xs):return dict(mean=statistics.mean(xs),p50=percentile(xs,50),p90=percentile(xs,90),p99=percentile(xs,99),max=max(xs))
def predictors(e,phase='pair'):
 phases=('dispatch','combine') if phase=='pair' else (phase,)
 matrix=[[sum(e[p+'_send'][s][d] for p in phases)*4096 for d in range(4)] for s in range(4)]
 outgoing=[sum(v for d,v in enumerate(row) if d!=s) for s,row in enumerate(matrix)]
 fans=[sum(v>0 for d,v in enumerate(row) if d!=s) for s,row in enumerate(matrix)]
 return dict(total_peer_bytes=sum(outgoing),self_bytes=sum(matrix[r][r] for r in range(4)),active_remote_ordered_pairs=sum(fans),max_rank_fanout=max(fans),max_rank_remote_bytes=max(outgoing))

def extract(batch):
 manifest=json.loads((PACKET/'batch_comm_prompt_provenance.json').read_text())
 if batch==8:
  recs=manifest['B8_source_receipts'];source=json.loads((PACKET/'trace_comm_counts.json').read_text());validate_events(source['events'],8)
  for r in recs:assert sha(r['path'])==r['sha256']
  return source,dict(status='PASS',local_batch=8,reused=True,raw_receipts=recs,counts_sha256=sha(PACKET/'trace_comm_counts.json'))
 folder=ROOT/f'capture{batch}';status=json.loads((folder/'status.json').read_text());assert status['status']=='PASS'
 ranks=[];receipts=[];reference=None;tokens=[]
 for rank in range(4):
  path=folder/f'rank{rank}.json';data=json.loads(path.read_text());assert data['status']=='PASS' and data['rank']==rank and data['local_batch']==batch
  assert data['prompts']==manifest['batches'][str(batch)]['ranks'][rank]
  assert data['prompt_manifest_sha256']==sha(PACKET/'batch_comm_prompt_provenance.json')
  cfg=data['config'];assert cfg['global_cache_ratio']==.3 and cfg['seed']==42 and cfg['admission']=='random' and cfg['eviction']=='lru' and not cfg['substitution_enabled']
  assert data['transport_env']=={'NCCL_CUMEM_ENABLE':'0'} and data['plan_cache_equal_all_ranks']
  assert len(data['events'])==432 and len(data['generated_tokens'])==batch and all(len(row)==9 and all(isinstance(x,int) and x>=0 for x in row) for row in data['generated_tokens'])
  hashes=[];rankrows=[]
  for index,e in enumerate(data['events']):
   assert e['event']==index and e['layer']==index%48 and e['step']==index//48
   owner={int(k):v for k,v in e['owner_by_expert'].items()}
   assert set(owner)=={x for row in e['raw_selected_experts'] for x in row}
   assert all(isinstance(v,int) and 0<=v<4 for v in owner.values())
   if index>=48:assert len(e['origin_ranks'])==4*batch and [e['origin_ranks'].count(r) for r in range(4)]==[batch]*4
   assert len(e['origin_ranks'])==len(e['raw_selected_experts'])==len(e['routing_weights'])==len(e['effective_token_routes'])
   exp_d=[0]*4;exp_c=[0]*4
   for origin,selected,weights,effective in zip(e['origin_ranks'],e['raw_selected_experts'],e['routing_weights'],e['effective_token_routes']):
    assert len(selected)==len(set(selected))==8
    assert {int(k):v for k,v in effective.items()}==dict(zip(selected,weights))
    dsts=[owner[x] for x in selected]
    if origin==rank:
     for dst in set(dsts):exp_d[dst]+=1
    exp_c[origin]+=dsts.count(rank)
   assert exp_d==e['dispatch_send_counts']==e['expected_dispatch']
   assert exp_c==e['return_send_counts']==e['expected_return']
   assert e['dispatch_row_bytes']==e['return_row_bytes']==4096
   hashes.append((e['plan_sha256'],e['cache_sha256'],digest([e['origin_ranks'],e['raw_selected_experts'],e['routing_weights']])))
   rankrows.append({k:e[k] for k in ('dispatch_send_counts','dispatch_recv_counts','return_send_counts','return_recv_counts')})
  if reference is None:reference=hashes
  else:assert hashes==reference
  ranks.append(rankrows);tokens.append(data['generated_tokens']);receipts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
  del data
 events=[]
 for i in range(432):
  e=dict(source_event=i,layer=i%48,step=i//48)
  for phase,original in (('dispatch','dispatch'),('combine','return')):
   for direction in ('send','recv'):e[phase+'_'+direction]=[ranks[r][i][original+'_'+direction+'_counts'] for r in range(4)]
   assert all(e[phase+'_send'][s][d]==e[phase+'_recv'][d][s] for s in range(4) for d in range(4))
  events.append(e)
 events=events[48:];validate_events(events,batch)
 totals={phase:sum(predictors(e,phase)['total_peer_bytes'] for e in events) for phase in ('dispatch','combine')}
 source=dict(status='PASS',local_batch=batch,world_size=4,physical_gpus=[0,1,4,5],hidden_size=2048,row_bytes=4096,dtype='bfloat16',raw_receipts=receipts,events=events,peer_bytes_per_trace=totals,total_peer_bytes_per_trace=sum(totals.values()))
 write(ROOT/f'counts_B{batch}.json',source)
 samples=status['memory']
 receipt=dict(status='PASS',local_batch=batch,reused=False,raw_receipts=receipts,counts_path=str(ROOT/f'counts_B{batch}.json'),counts_sha256=sha(ROOT/f'counts_B{batch}.json'),
  decode_events=384,all_events_validated=432,plan_cache_route_hashes_equal_all_ranks=True,exact_only=True,replication=False,
  generated_tokens_sha256=digest(tokens),cache_ratio=.3,model_timing_evidence=False,
  memory=dict(peak_rss_gib=max(s.get('group_rss_bytes',0) for s in samples)/2**30,min_host_available_gib=min(s['host_available_bytes'] for s in samples)/2**30,min_gpu_free_mib=min(g['free_mib'] for s in samples for g in s['gpu'].values())))
 write(PACKET/f'batch_comm_capture_B{batch}.json',receipt)
 return source,receipt

def geometry(sources):
 rows=[];per_event=[]
 for batch,source in sources.items():
  for phase in ('dispatch','combine','union'):
   phases=('dispatch','combine') if phase=='union' else (phase,)
   sizes=[e[p+'_send'][s][d]*4096 for e in source['events'] for p in phases for s in range(4) for d in range(4) if s!=d and e[p+'_send'][s][d]>0]
   preds=[predictors(e,'pair' if phase=='union' else phase) for e in source['events']]
   total=sum(x['total_peer_bytes'] for x in preds);selfbytes=sum(x['self_bytes'] for x in preds)
   row=dict(local_batch=batch,global_batch=4*batch,phase=phase,peer_bytes=total,self_bytes=selfbytes,self_byte_fraction=selfbytes/(total+selfbytes),nonzero_remote_messages=len(sizes))
   row.update({k+'_message_bytes':v for k,v in summary(sizes).items()})
   for key in ('active_remote_ordered_pairs','max_rank_fanout','max_rank_remote_bytes'):
    row.update({key+'_'+k:v for k,v in summary([x[key] for x in preds]).items()})
   rows.append(row)
   for i,p in enumerate(preds):per_event.append(dict(local_batch=batch,phase=phase,source_event=i+48,**p))
 csvwrite('batch_comm_geometry.csv',rows)
 write(PACKET/'batch_comm_geometry.json',dict(status='PASS',rows=rows,per_event=per_event,union_note='Message distribution concatenates dispatch+combine; union fan-out counts unique rank-pair edges.'))
 return rows

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--batches',type=int,nargs='+',required=True);a=p.parse_args()
 sources={};receipts={}
 for b in a.batches:sources[b],receipts[b]=extract(b)
 geometry(sources)
 print(json.dumps({b:dict(status=r['status'],reused=r['reused'],memory=r.get('memory')) for b,r in receipts.items()},indent=2))
