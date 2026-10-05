"""Offline Stage B: freeze A-only calibration, reconstruct and validate old captures."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import argparse,bisect,csv,hashlib,json,sqlite3,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr
from prepare_critical_microbench import P,PACKET,ROOT,OLD,write,features
from analyze_refactor_nsys import duration
BROOT=ROOT/'stage_b'
POLICIES=('BR','OLD_CA','FCA','LA_CA')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p,sources):sources[str(p)]=sha(p);return json.loads(p.read_text())
def calibrate():
 sources={};a=read(PACKET/'A2A_SPLIT_RESULTS.json',sources)['rows'];tau=read(PACKET/'EXPERT_TAU_RESULTS.json',sources)['rows'];skew=read(PACKET/'MICROBENCH_RESULTS.json',sources)
 models={}
 for packet in (4120,4096):
  xs=[x for x in a if x['packet_bytes']==packet and x['shape'] in ('BALANCED','PAIR_HOT','DEST_HOT','SRC_HOT')]
  # Smallest concentration-sensitive model, selected before opening validation:
  # nonnegative intercept and critical-rank incident traffic slope.
  X=np.array([[1,x['max_incident']] for x in xs]);y=np.array([x['median_max_local_completion_ms'] for x in xs]);coef,_=nnls(X,y);pred=X@coef
  models[str(packet)]=dict(features=['intercept','max_incident_packets'],coefficients_ms=coef.tolist(),a1_training_ids=[x['id'] for x in xs],calibration_median_absolute_relative_error=float(np.median(abs(pred-y)/y)),observed_total_range=[min(x['total_packets'] for x in xs),max(x['total_packets'] for x in xs)],max_incident_range=[min(x['max_incident'] for x in xs),max(x['max_incident'] for x in xs)])
 primary=[x for x in tau if x['primary_ladder']];assert all(x['stable'] for x in primary)
 ns=sorted({x['rows'] for x in primary});times=[float(np.median([x['median_ms'] for x in primary if x['rows']==n])) for n in ns]
 alpha=float(np.dot(ns,times)/np.dot(ns,ns))
 model=dict(status='FROZEN_BEFORE_VALIDATION',plan_commit='5d9d2b2ae1a740ffbad1288a2e67a08c7f814446',transport=models,tau=dict(rows=ns,ms=times,interpolation='linear in rows; no extrapolation above512; tau(0)=0',source='median of four rank medians, stable primary power ladder only; unstable supplemental n18 not used'),row_only_ms_per_row=alpha,arrival_slope_a2=skew['arrival_skew_tail_slope'],layer_formula='F(S)+max(sum_e tau(n_e))+R(S.T)',return_residency_formula='R(S.T)+max(ready)-ready[r]',arrival_coefficient_used=1.,source_sha256=sources,ordering_rule='within each cache: BR and LA_CA within2% of their pair mean; both below OLD_CA, OLD_CA below FCA',gate_scope='all eight held-out cells must meet expert/layer correlations and phase aggregate errors; no TPOT-fitting; aggregate and per-cell decisions retained',observed_critical_target='max over ranks of rank-local elapsed from first forward NCCL kernel start to last return NCCL kernel end for the SAME causal event. Contains unmodeled host/H2D/pack gaps; not a sum of phase maxima and not primary TPOT.')
 write(PACKET/'CRITICAL_PATH_MODEL.json',model);BROOT.mkdir(exist_ok=True);write(BROOT/'frozen_model.json',model)
 print('Model frozen:',models,flush=True)
def gpu_spans(dbpath):
 """Use the existing CUPTI->NVTX causal association; never adjacent host bins."""
 db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 strings=dict(db.execute('select id,value from StringIds'));ranges=defaultdict(list);ids={}
 for r in db.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end>start'):
  name=strings.get(r['textId'],r['text'])
  if not name:continue
  if name.startswith('decode.event.'):
   ids[(r['globalTid'],r['start'])]=int(name.rsplit('.',1)[1]);name='decode.event'
  if name in ('decode.event','moe.forward_a2a','moe.expert_compute','moe.return_a2a'):ranges[(r['globalTid'],name)].append((r['start'],r['end']))
 for k,v in list(ranges.items()):v.sort();ranges[k]=([a for a,b in v],v)
 def inside(tid,name,a,b):
  pair=ranges.get((tid,name))
  if not pair:return False
  i=bisect.bisect_right(pair[0],a)-1
  return i>=0 and pair[1][i][1]>=b
 tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")};runtime={}
 for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
  if table in tables:
   for r in db.execute(f'select start,end,globalTid,correlationId from {table}'):
    runtime.setdefault((r['globalTid'] & 0xFFFFFFFFFF000000,r['correlationId']),(r['start'],r['end'],r['globalTid']))
 phases=defaultdict(lambda:defaultdict(list))
 for r in db.execute('select start,end,globalPid,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
  api=runtime.get((r['globalPid'],r['correlationId']))
  if not api:continue
  a,b,tid=api
  if not inside(tid,'decode.event',a,b):continue
  starts,_=ranges[(tid,'decode.event')];event=ids[(tid,starts[bisect.bisect_right(starts,a)-1])]
  for name in ('moe.forward_a2a','moe.expert_compute','moe.return_a2a'):
   if inside(tid,name,a,b) and (name=='moe.expert_compute' or 'nccl' in strings.get(r['demangledName'],'').lower()):phases[event][name].append((r['start'],r['end']))
 result={}
 for event,x in phases.items():
  assert all(k in x for k in ('moe.forward_a2a','moe.expert_compute','moe.return_a2a'))
  start=min(a for a,b in x['moe.forward_a2a']);end=max(b for a,b in x['moe.return_a2a'])
  result[event]=dict(critical_span_ms=(end-start)/1e6,forward_start_ns=start,return_end_ns=end,expert_start_ns=min(a for a,b in x['moe.expert_compute']),expert_end_ns=max(b for a,b in x['moe.expert_compute']),return_start_ns=min(a for a,b in x['moe.return_a2a']),union_ms={k:duration(v)/1e6 for k,v in x.items()})
 assert len(result)==3072;db.close();return result

def cell(cache,policy):
 from mgo_v2.controller import DecodePrefetchController
 from mgo_v2.predictor import TransitionPredictor
 import psutil
 assert psutil.virtual_memory().available>256*2**30
 started=time.time();sources={};model=read(BROOT/'frozen_model.json',sources);inp=OLD/cache/'inputs_B128_H64'
 meta=read(inp/'input_receipt.json',sources)
 a={k:np.load(inp/f'{k}.npy',mmap_mode='r') for k in ('selected','weights','offsets','prefill_origins','decode_origins','gates')}
 for k in a:sources[str(inp/f'{k}.npy')]=sha(inp/f'{k}.npy')
 predictor=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/predictor/transition.npy');sources[str(predictor)]=sha(predictor)
 c=DecodePrefetchController([461]*4 if cache=='C30' else [922,922,921,921],2,policy,meta['placement_seed'],TransitionPredictor(np.load(predictor)))
 capture=OLD/cache/f'POLICY_REGIME_PROFILE_{cache}_NSYS2025_V3_OPT_PF_OVERLAP_{policy}_B128_H64'
 comm=[read(capture/f'communication_rank{r}.json',sources)['events'] for r in range(4)]
 intervals=[read(capture/f'interval_analysis/rank{r}_intervals.json',sources) for r in range(4)]
 spans=[]
 for r in range(4):
  db=capture/f'interval_analysis/rank{r}.sqlite';sources[str(db)]=sha(db);spans.append(gpu_spans(db))
 rows=[]
 def transport(S,packet):
  co=model['transport'][str(packet)]['coefficients_ms'];return float(co[0]+co[1]*max(S.sum(0)+S.sum(1)))
 for event in range(3120):
  lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];sel=a['selected'][lo:hi]
  out,_,_=c.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
  if event>=48:
   _,effective,masses,lengths,dest,fetches,_=out;assert np.all(lengths==8)
   n=np.bincount(sel.ravel(),minlength=128);experts=np.flatnonzero(n);owners=c.main.primary[event%48*128+experts];assert np.all((owners>=0)&(owners<4))
   S=np.zeros((4,4),int)
   for src in range(4):
    ds=dest[org==src]
    for dst in range(4):S[src,dst]=int(np.any(ds==dst,axis=1).sum())
   for rank in range(4):
    cr=comm[rank][event-48];assert cr['event']==event
    assert cr['send_token_rows']==S[rank].tolist() and cr['recv_token_rows']==S[:,rank].tolist(),('split mismatch',cache,policy,event,rank)
    assert int(n[experts[owners==rank]].sum())==cr['expert_rows']
    assert sum(f[0]==rank for f in fetches)==cr['mandatory_fetches']
   np.fill_diagonal(S,0)
   lists=[[[int(e),int(n[e])] for e in experts[owners==r]] for r in range(4)]
   ready=[float(sum(np.interp(nn,model['tau']['rows'],model['tau']['ms']) for e,nn in es)) for es in lists]
   rawrows=[sum(nn for e,nn in es) for es in lists];rowready=np.array(rawrows)*model['row_only_ms_per_row']
   fwd=transport(S,4120);ret=transport(S.T,4096)
   observed={name:[intervals[r]['interval_decomposition']['causal_event_kernel_union_ms'][str(event)][name] for r in range(4)] for name in ('moe.forward_a2a.nccl','moe.expert_compute','moe.return_a2a.nccl')}
   for r in range(4):
    for name,key in [('moe.forward_a2a','moe.forward_a2a.nccl'),('moe.expert_compute','moe.expert_compute'),('moe.return_a2a','moe.return_a2a.nccl')]:assert abs(spans[r][event]['union_ms'][name]-observed[key][r])<1e-8
   rows.append(dict(cache=cache,policy=policy,event=event,split=S.tolist(),**features(S),executing_expert_rows_by_rank=lists,current_active_owners=dict(zip(map(str,experts),map(int,owners))),rank_rows=rawrows,pred_forward_ms=fwd,pred_expert_rank_ms=ready,pred_expert_max_ms=max(ready),pred_return_rank_ms=[ret+max(ready)-t for t in ready],pred_return_base_ms=ret,pred_critical_ms=fwd+max(ready)+ret,pred_row_expert_max_ms=float(max(rowready)),pred_row_critical_ms=float(fwd+max(rowready)+ret),obs_forward_rank_ms=observed['moe.forward_a2a.nccl'],obs_forward_max_ms=max(observed['moe.forward_a2a.nccl']),obs_expert_rank_ms=observed['moe.expert_compute'],obs_expert_max_ms=max(observed['moe.expert_compute']),obs_return_rank_ms=observed['moe.return_a2a.nccl'],obs_critical_ms=max(spans[r][event]['critical_span_ms'] for r in range(4)),rank_gpu_spans=[spans[r][event] for r in range(4)],h2d_by_source_event_rank_ms=[intervals[r]['H2D_by_source_event_ms'].get(str(event),0.) for r in range(4)],mandatory_fetches_by_rank=[sum(f[0]==r for f in fetches) for r in range(4)],transport_volume_out_of_calibration=not(1389<=S.sum()<=1528)))
   hist=np.bincount((sel.astype(np.int64)+org.astype(np.int64)[:,None]*128).ravel(),minlength=512).reshape(4,128);c.plan_prefetch_next(hist)
 c.arena.assert_consistent();assert not c.pending
 proof=read(inp/f'{policy}_P2_proof.json',sources)
 assert c.counters==proof['counters']
 for r,cap in enumerate(c.main.capacities):assert hashlib.sha256(c.main.slots[r,:cap].tobytes()).hexdigest()==proof['rank_state_hashes'][r]
 path=BROOT/f'{cache}_{policy}.json';write(path,dict(status='PASS',rows=rows,source_sha256=sources,seconds=time.time()-started,rss_bytes=psutil.Process().memory_info().rss,controller_counters=c.counters))
 print(cache,policy,'PASS',round(time.time()-started,1),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--calibrate',action='store_true');ap.add_argument('--cell',nargs=2);args=ap.parse_args()
 if args.calibrate:calibrate()
 elif args.cell:cell(*args.cell)
