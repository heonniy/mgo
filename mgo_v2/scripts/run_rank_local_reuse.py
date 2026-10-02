#!/usr/bin/env python3
"""Bounded single-process CPU-only rank-local reuse study."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import argparse,csv,gzip,hashlib,json,math,statistics,sys,time
from pathlib import Path
import numpy as np
from replica_pareto_cpu import ReplicaReplay,IndependentZeroReplay,ROW_BYTES,EXPERT_BYTES
from rank_local_reuse import *
PACKET=Path(__file__).resolve().parents[1]/'experiments/rank_local_reuse_oracle_20261003'
OLD=PACKET.parent/'fetch_comm_pareto_p2p_20261002'
ROOT=Path('/home/hwlee/mgo-results/rank_local_reuse_oracle_20261003')
CAPS=[461,461,461,460]
FIELDS='first_copy_fetches reload_fetches replica_fetches total_fetches expert_h2d_bytes peer_activation_bytes dispatch_bytes combine_bytes remote_token_rank_pairs remote_expert_routes'.split()

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def receipt(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def csvfile(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def aggregate(rows):
 return dict(**{k:sum(r[k] for r in rows) for k in FIELDS},events=len(rows),
  duplicate_slot_mean=statistics.mean(r['duplicate_slots'] for r in rows),duplicate_slot_peak=max(r['duplicate_peak_in_event'] for r in rows),unique_resident_experts_mean=statistics.mean(r['unique_resident_experts'] for r in rows))
def source(batch):
 path=OLD/('exact_payload_capture_summary.json' if batch==8 else f'batch_comm_capture_B{batch}.json')
 data=json.loads(path.read_text());assert data['status']=='PASS'
 assert data['plan_cache_equal_all_ranks'] if batch==8 else data['plan_cache_route_hashes_equal_all_ranks']
 for r in data['raw_receipts']:
  assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
 raw=json.loads(Path(data['raw_receipts'][0]['path']).read_text());assert raw['status']=='PASS' and len(raw['events'])==432
 events=[];demand={}
 for i,e in enumerate(raw['events']):
  assert (e['event'],e['layer'],e['step'])==(i,i%48,i//48)
  origins=np.asarray(e['origin_ranks'],dtype=np.int64);selected=np.asarray(e['raw_selected_experts'],dtype=np.int64)
  assert selected.shape==(len(origins),8) and np.all((selected>=0)&(selected<128))
  if i>=48:assert np.array_equal(np.bincount(origins,minlength=4),[batch]*4)
  events.append((e['layer'],origins,selected))
  for expert in np.unique(selected):demand.setdefault((e['layer'],int(expert)),[]).append(i)
 del raw
 return events,demand,dict(summary=receipt(path),raw_receipts=data['raw_receipts'],cross_rank_validation=True)

def concentration(values,fraction):
 n=max(1,math.ceil(len(values)*fraction));return sum(sorted(values,reverse=True)[:n])/sum(values)
def gini(values):
 x=sorted(values);n=len(x);return 2*sum((i+1)*v for i,v in enumerate(x))/(n*sum(x))-(n+1)/n

def r02(batch):
 started=time.monotonic();events,_,provenance=source(batch)
 replay=ReplicaReplay(CAPS,0);reference=IndependentZeroReplay(CAPS);rows=[];timelines={}
 archive=ROOT/f'F_B{batch}_events.jsonl.gz'
 with gzip.open(archive,'wt') as out:
  for i,(layer,origins,selected) in enumerate(events):
   before=replay.state() if i>=48 else None
   row,dest,tr,ops=replay.event(layer,origins,selected)
   state,rd,sd,sc,first,reloads=reference.event(layer,origins,selected)
   assert replay.state()==state and np.array_equal(dest,rd) and np.array_equal(tr['dispatch'],sd) and np.array_equal(tr['combine'],sc)
   assert (row['first_copy_fetches'],row['reload_fetches'])==(first,reloads) and row['replica_fetches']==row['duplicate_slots']==0
   rows.append(row)
   if i>=48:
    out.write(json.dumps(dict(event=i,step=i//48,layer=layer,origins=origins.tolist(),selected=selected.tolist(),destinations=dest.tolist(),remote=(dest!=origins[:,None]).tolist(),dispatch=tr['dispatch'].tolist(),dispatch_recv=tr['dispatch'].T.tolist(),combine=tr['combine'].tolist(),combine_recv=tr['combine'].T.tolist(),cache_before=before,cache_after=state),separators=(',',':'))+'\n')
    for (expert,rank),record in marginal_pairs(origins,selected,dest).items():
     line=timelines.setdefault((layer,expert,rank),{k:[0]*9 for k in record})
     for k,v in record.items():line[k][i//48]=v
   if (i+1)%48==0:print(json.dumps(dict(stage='R0',batch=batch,events=i+1,seconds=round(time.monotonic()-started,2))),flush=True)
 baseline=aggregate(rows[48:])
 if batch==8:
  expected=json.loads((PACKET/'matrix.json').read_text())['baseline_B8_decode']['F']
  assert all(baseline[k]==v for k,v in expected.items()),(baseline,expected)
  old=json.loads((OLD/'replica_pareto_screen.json').read_text())['points'][0]
  assert all(baseline[k]==old['decode'][k] for k in FIELDS)
 timelines={k:v for k,v in timelines.items() if sum(v['remote_routes'])}
 compact=[dict(layer=k[0],expert=k[1],rank=k[2],**v) for k,v in sorted(timelines.items())]
 timeline_path=ROOT/f'timelines_B{batch}.json';timeline_path.write_text(json.dumps(compact,separators=(',',':'))+'\n')
 pair_rows=[];occurrence_path=ROOT/f'reuse_occurrences_B{batch}.jsonl.gz';upper_path=ROOT/f'persistence_candidates_B{batch}.jsonl.gz';upper=[]
 total_marginal=sum(sum(v['marginal_bytes']) for v in timelines.values());within={h:0 for h in (1,2,4)}
 with gzip.open(occurrence_path,'wt') as fo,gzip.open(upper_path,'wt') as fu:
  for key,line in sorted(timelines.items()):
   active=[s for s in range(1,9) if line['remote_routes'][s]>0];first=active[0]
   pair=dict(batch=batch,layer=key[0],expert=key[1],rank=key[2],remote_routes=sum(line['remote_routes']),marginal_peer_bytes=sum(line['marginal_bytes']),remote_steps=len(active),first_remote_step=first,max_remote_burst=max(burst(line['remote_routes'],s) for s in active),remote_steps_mask=''.join('1' if line['remote_routes'][s] else '0' for s in range(1,9)))
   pair_rows.append(pair)
   for s in active:
    distance=next_distance(line['remote_routes'],s)
    record=dict(batch=batch,layer=key[0],expert=key[1],rank=key[2],step=s,remote_routes=line['remote_routes'][s],marginal_peer_bytes=line['marginal_bytes'][s],next_remote_reuse_distance=distance,remote_burst=burst(line['remote_routes'],s),demand_burst=burst(line['demand_routes'],s),horizons={str(h):future_stats(line,s,h) for h in HORIZONS})
    fo.write(json.dumps(record,separators=(',',':'))+'\n')
    for h in within:
     if distance is not None and distance<=h:within[h]+=line['marginal_bytes'][s]
   break_even=next((h for h in range(1,9-first) if future_stats(line,first,h)['future_peer_bytes_saved']>=EXPERT_BYTES),None)
   for h in HORIZONS:
    row=dict(batch=batch,layer=key[0],expert=key[1],rank=key[2],first_remote_step=first,horizon=h,**future_stats(line,first,h),replica_fetch_bytes=EXPERT_BYTES,first_break_even_horizon_by_bytes=break_even)
    fu.write(json.dumps(row,separators=(',',':'))+'\n');upper.append(row)
 values={k:[r[k] for r in pair_rows] for k in ('remote_routes','marginal_peer_bytes')}
 conc={k:{str(f):concentration(v,f) for f in (.01,.05,.1,.2)} for k,v in values.items()}
 reuse={str(h):within[h]/total_marginal for h in within}
 gate=conc['marginal_peer_bytes']['0.1']>=.4 or reuse['4']>=.25
 summaries=[]
 for h in HORIZONS:
  ur=[r for r in upper if r['horizon']==h];savings=[r['future_peer_bytes_saved'] for r in ur]
  summaries.append(dict(batch=batch,horizon=h,candidates=len(ur),positive_saving_candidates=sum(x>0 for x in savings),sum_individual_future_peer_bytes=sum(savings),sum_hypothetical_fetch_bytes=len(ur)*EXPERT_BYTES,mean_future_peer_bytes=statistics.mean(savings),p50_future_peer_bytes=float(np.percentile(savings,50)),p90_future_peer_bytes=float(np.percentile(savings,90)),max_future_peer_bytes=max(savings),max_peer_per_fetch_byte_ratio=max(savings)/EXPERT_BYTES,total_reuse_steps=sum(r['reuse_steps'] for r in ur),total_reuse_routes=sum(r['reuse_routes'] for r in ur),break_even_candidates=sum(r['future_peer_bytes_saved']>=EXPERT_BYTES for r in ur)))
 csvfile(PACKET/f'reuse_pairs_B{batch}.csv',pair_rows)
 result=dict(status='PASS',batch=batch,baseline=baseline,independent_F_events=432,provenance=provenance,pair_count=len(pair_rows),remote_marginal_bytes=total_marginal,concentration=conc,gini={k:gini(v) for k,v in values.items()},marginal_byte_share_next_remote_reuse=reuse,R2_gate=gate,decision='CONTINUE_R3' if gate else 'LOW_RANK_LOCAL_REUSE',persistence=summaries,
  artifacts=[receipt(x) for x in (archive,timeline_path,occurrence_path,upper_path)],elapsed_seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 write(PACKET/f'R02_B{batch}.json',result)
 print(json.dumps({k:result[k] for k in ('batch','baseline','pair_count','concentration','gini','marginal_byte_share_next_remote_reuse','R2_gate','elapsed_seconds','peak_rss_mib')},indent=2),flush=True)

def r3(batch,penalty_mode):
 gate=json.loads((PACKET/'R02_B8.json').read_text());assert gate['R2_gate']
 r02data=json.loads((PACKET/f'R02_B{batch}.json').read_text())
 path=ROOT/f'timelines_B{batch}.json';assert sha(path)==next(r['sha256'] for r in r02data['artifacts'] if r['path']==str(path))
 timelines={(r['layer'],r['expert'],r['rank']):{k:r[k] for k in ('demand_routes','remote_routes','marginal_bytes')} for r in json.loads(path.read_text())}
 events,demand,provenance=source(batch);points=[];started=time.monotonic()
 for h in HORIZONS:
  for threshold in THRESHOLDS:
   rep=SelectiveReplay(CAPS,timelines,demand,h,threshold,penalty_mode);rows=[]
   for i,(layer,origins,selected) in enumerate(events):
    row,_,_,_=rep.step(i,layer,origins,selected);rows.append(row)
   totals=aggregate(rows[48:]);lives=rep.finish();assert totals['replica_fetches']==len(lives)
   life_path=ROOT/f'replica_lifetimes_B{batch}_H{h}_T{threshold}.jsonl.gz'
   with gzip.open(life_path,'wt') as f:
    for record in lives:f.write(json.dumps(record,separators=(',',':'))+'\n')
   point=dict(batch=batch,horizon=h,threshold_peer_bytes=threshold,penalty_mode=penalty_mode,**totals,
    replica_admissions=len(lives),reused_replicas=sum(r['reused'] for r in lives),fraction_reused_before_eviction=sum(r['reused'] for r in lives)/len(lives) if lives else 0,
    mean_replica_lifetime_layer_events=statistics.mean(r['lifetime_layer_events'] for r in lives) if lives else 0,
    mean_replica_lifetime_decode_steps=statistics.mean(r['lifetime_layer_events']/48 for r in lives) if lives else 0,
    right_censored_replicas=sum(r['right_censored'] for r in lives),victim_induced_reload_count=rep.victim_reloads,accepted_expected_victim_penalties=rep.accepted_victim_penalties,
    peer_bytes_saved_vs_F=r02data['baseline']['peer_activation_bytes']-totals['peer_activation_bytes'],
    peer_bytes_saved_per_replica_fetch=(r02data['baseline']['peer_activation_bytes']-totals['peer_activation_bytes'])/len(lives) if lives else 0,lifetimes=receipt(life_path))
   points.append(point);write(PACKET/f'R3_B{batch}.json',dict(status='RUNNING',batch=batch,points=points,provenance=provenance));print(json.dumps({k:point[k] for k in ('batch','horizon','threshold_peer_bytes','expert_h2d_bytes','peer_activation_bytes','replica_admissions')}),flush=True)
 write(PACKET/f'R3_B{batch}.json',dict(status='PASS',batch=batch,points=points,provenance=provenance,elapsed_seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['r02','r3']);parser.add_argument('--batch',type=int,choices=[8,16,32],required=True);parser.add_argument('--penalty-mode',choices=['fetch_normalized','subtract_expert_bytes'],default='fetch_normalized');a=parser.parse_args()
 ROOT.mkdir(exist_ok=True)
 target=PACKET/f'{"R02" if a.stage=="r02" else "R3"}_B{a.batch}.json';assert not target.exists(),'Do not repeat completed/failed cells without an explicit amendment'
 try:
  if a.stage=='r02':r02(a.batch)
  else:r3(a.batch,a.penalty_mode)
  assert 'torch' not in sys.modules and os.environ['CUDA_VISIBLE_DEVICES']==''
 except BaseException as exc:
  write(PACKET/f'failure_{a.stage}_B{a.batch}.json',dict(status='FAIL',error=repr(exc)));raise
