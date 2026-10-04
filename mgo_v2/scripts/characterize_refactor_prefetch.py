"""Bounded process-parallel CPU characterization of shared C+P semantics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib,itertools,concurrent.futures
from pathlib import Path
import numpy as np
import psutil
from mgo_v2.prefetch import PrefetchShadow
from mgo_v2.predictor import TransitionPredictor
from la_placement import load_assignment,estimate_loads
from br_carep_cpu import balanced_assignment
from calibrate_refactor_predictor import histogram
from ca_stress_common import write
P=Path(__file__).resolve().parents[1];PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')

def job(spec):
 batch,policy,budget=spec;out=ROOT/'cpu_characterization'/f'B{batch}_{policy}_P{budget}.json'
 if out.exists():return json.loads(out.read_text())
 assert psutil.virtual_memory().available>256*2**30
 raw=Path('/home/hwlee/mgo-results/la_physical_validation_20261004')/f'B{batch}';source=json.loads((raw/'receipt.json').read_text())
 a={k:np.load(raw/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
 predictor=TransitionPredictor(np.load(ROOT/'predictor/transition.npy'));shadow=PrefetchShadow([3686//8+(r<3686%8) for r in range(8)],budget,policy,source['winner']['placement_seed'],predictor)
 critical=0;prefcounts=np.zeros(8,np.int64);ca_regret=0;la_regret=0;regret_events=0;start=time.time()
 for event in range(12336):
  if event%768==0:assert psutil.virtual_memory().available>128*2**30
  lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];sel=a['selected'][lo:hi];h=histogram(sel,org)
  if shadow.pending:
   keys=sorted(shadow.pending);experts=np.array([k%128 for k in keys],np.int64);ranks=np.array([shadow.pending[k][0] for k in keys],np.int64);demand=h.T.copy()
   oracle_ca=balanced_assignment(demand,experts,8,False);ca_regret+=sum(int(demand[e,r])-int(demand[e,s]) for e,r,s in zip(experts,oracle_ca,ranks))
   oracle_la=load_assignment(demand,experts,shadow.main.owner,event%48);loads=estimate_loads(demand,shadow.main.owner,event%48);actual=loads.copy();ideal=loads.copy()
   for e,r,s in zip(experts,ranks,oracle_la):actual[r]+=int(demand[e].sum());ideal[s]+=int(demand[e].sum())
   la_regret+=max(0,int(actual.max()-ideal.max()));regret_events+=1
  result,promotions,discard=shadow.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
  if event>=48:critical+=int(np.bincount(result[4].ravel(),minlength=8).max())
  for rank,key,slot in shadow.plan_prefetch_next(h):prefcounts[rank]+=1
 assert not shadow.pending
 m=shadow.main;digest=hashlib.sha256()
 for key in ['slots','owner','primary']:digest.update(getattr(m,key).tobytes())
 old=json.loads((P/'experiments/r8_real_replica_batch_scaling_20261004/RESULT.json').read_text());baseline=next(x for x in old['selected_full'][f'B{batch}_LOAD'] if x['policy']==policy)
 if budget==0:
  assert digest.hexdigest()==baseline['state_sha256'];assert critical==baseline['decode']['critical_rows_sum'];assert shadow.counters['mandatory']*9437184==baseline['moe_total']['H2D_bytes']
 row=dict(status='PASS',batch=batch,policy=policy,P=budget,seconds=time.time()-start,counters=shadow.counters,prefetch_per_rank=prefcounts.tolist(),critical_rows_sum=critical,baseline_critical_rows=baseline['decode']['critical_rows_sum'],baseline_H2D_bytes=baseline['moe_total']['H2D_bytes'],modeled_total_H2D_bytes=(shadow.counters['mandatory']+shadow.counters['issued'])*9437184,CA_local_route_regret=ca_regret,LA_max_rank_row_regret=la_regret,regret_events=regret_events,final_state_sha256=digest.hexdigest(),P0_archived_parity=budget==0,extra_HBM_per_rank=budget*9437184)
 write(out,row);return row

def main():
 (ROOT/'cpu_characterization').mkdir(exist_ok=True)
 specs=list(itertools.product((128,256),('BR','CA','LA'),(0,1,2,4,8)));rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for row in pool.map(job,specs):
   rows.append(row);print(json.dumps({k:row[k] for k in ['batch','policy','P','seconds']}),flush=True)
 for row in rows:
  c=row['counters'];row['useful_fraction']=c['useful']/max(1,c['issued']);row['H2D_growth']=row['modeled_total_H2D_bytes']/row['baseline_H2D_bytes']-1
 # Keep at most three BR-only budget candidates; include a small budget and the best
 # traffic/savings knee. Physical selection remains mandatory, and is not inferred here.
 br=[r for r in rows if r['policy']=='BR' and r['P']>0]
 scores={p:sum(r['H2D_growth'] for r in br if r['P']==p) for p in (1,2,4,8)}
 bounded=[p for p in scores if max(r['H2D_growth'] for r in br if r['P']==p)<=.12]
 candidate_set=sorted(set([1]+sorted(scores,key=scores.get)[:2]+([max(bounded)] if bounded else [])))
 write(PACKET/'M3_cpu_characterization.json',dict(status='PASS',rows=rows,physical_budget_candidates=candidate_set,selection='BR-only shortlist: two lowest traffic budgets plus largest budget with <=12% H2D growth on both batches, retaining a stronger-overlap candidate; no runtime winner declared',worker_limit=4,host_reserve_GiB=128,semantics='Instant-ready CPU prefetch upper bound; all issued copies charged including unused predictions. Runtime queue cancellation/readiness is evaluated physically later.',candidate_fairness='Same predictor, ranking and post-current-MAIN filter for all policies; candidate identity is equal given equal cache state. Different policy cache trajectories can change the filtered set.',owner_regret='Actual next-layer demand is used only by offline evaluation at target arrival, never by prediction/placement. Regret compares fixed candidate sets under balanced quotas.'))
 print('PASS M3',flush=True)
if __name__=='__main__':main()
