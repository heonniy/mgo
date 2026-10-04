"""Disjoint ShareGPT calibration and held-out MATH LOAD quality measurement."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,time
from pathlib import Path
import numpy as np
from mgo_v2.predictor import transition_counts,normalize,TransitionPredictor,choose_candidates
from env_offload_policy import Policy
from ca_stress_common import write,sha
P=Path(__file__).resolve().parents[1];PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004');SOURCE=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004')

def input_hashes(ds):
 rows=json.loads((SOURCE/(ds+'_requests.json')).read_text())['requests']
 return {hashlib.sha256(json.dumps(x['input_ids'],separators=(',',':')).encode()).hexdigest() for x in rows}
def histogram(selected,origins):
 return np.bincount((selected.astype(np.int64)+origins.astype(np.int64)[:,None]*128).ravel(),minlength=1024).reshape(8,128)
def main():
 root=ROOT/'predictor';root.mkdir(parents=True,exist_ok=True)
 assert not input_hashes('MATH')&input_hashes('ShareGPT')
 source=SOURCE/'pool/ShareGPT/decode_selected.npy';routes=np.load(source,mmap_mode='r');counts=transition_counts(routes);table=normalize(counts)
 assert np.array_equal(counts,transition_counts(routes));np.save(root/'transition_counts.npy',counts);np.save(root/'transition.npy',table);pred=TransitionPredictor(table)
 quality=[]
 for batch in (128,256):
  raw=Path('/home/hwlee/mgo-results/la_physical_validation_20261004')/f'B{batch}';receipt=json.loads((raw/'receipt.json').read_text())
  a={k:np.load(raw/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
  cap=np.array([3686//8+(r<3686%8) for r in range(8)],np.int32);pol=Policy(cap,np.zeros((48,128,128),np.float32),False,0,receipt['winner']['placement_seed'])
  acc={p:dict(issued=0,active_hits=0,miss_hits=0,actual_active=0,actual_misses=0,demand_hit=0,demand_total=0,oracle_miss_hits=0,events=0) for p in (1,2,4,8)};mae_sum=0.;mae_n=0
  for event in range(12336):
   lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];sel=a['selected'][lo:hi]
   pol.apply(event,sel,a['weights'][lo:hi],org,a['gates'][event],np.zeros((128,8),np.int32))
   layer=event%48
   if event<48 or layer==47:continue
   predicted=pred.predict_next(layer,histogram(sel,org));nlo,nhi=a['offsets'][event+1:event+3];actual=histogram(a['selected'][nlo:nhi],a['decode_origins']);totals=actual.sum(0);active=totals>0
   resident=pol.owner[(layer+1)*128:(layer+2)*128]!=0;miss=active&~resident;mae_sum+=float(np.abs(predicted*8-actual).sum());mae_n+=actual.size
   for budget,z in acc.items():
    chosen=choose_candidates(predicted,pol.slots.ravel(),[],layer+1,budget)
    z['issued']+=len(chosen);z['active_hits']+=int(active[chosen].sum());z['miss_hits']+=int(miss[chosen].sum());z['actual_active']+=int(active.sum());z['actual_misses']+=int(miss.sum());z['demand_hit']+=int(totals[chosen].sum());z['demand_total']+=int(totals[miss].sum());z['oracle_miss_hits']+=min(len(chosen),int(miss.sum()));z['events']+=1
   if event%1536==0:print(json.dumps(dict(stage='predictor-eval',batch=batch,event=event)),flush=True)
  for budget,z in acc.items():quality.append(dict(batch=batch,P=budget,precision=z['active_hits']/max(1,z['issued']),recall=z['active_hits']/max(1,z['actual_active']),actual_miss_recall=z['miss_hits']/max(1,z['actual_misses']),demand_weighted_miss_recall=z['demand_hit']/max(1,z['demand_total']),oracle_miss_recall=z['oracle_miss_hits']/max(1,z['actual_misses']),predicted_route_rows_MAE=mae_sum/mae_n,counters=z))
 out=dict(status='PASS',calibration=dict(dataset='ShareGPT',requests=2048,decode=256,source_sha256=sha(source),table_sha256=sha(root/'transition.npy'),input_hashes_sha256=hashlib.sha256('\n'.join(sorted(input_hashes('ShareGPT'))).encode()).hexdigest()),evaluation=dict(dataset='MATH',batches=[128,256],input_token_overlap=0),deterministic_double_count=True,quality=quality,prediction_units='Plan Dhat=h@T/top_k is a normalized token score. Multiply by top_k for route-row MAE; candidate ranking is unchanged.',owner_regret='Deferred to policy-conditioned M3 placement replay; this stage evaluates BR cache-shadow quality only.',table_path=str(root/'transition.npy'))
 write(PACKET/'M2_predictor_quality.json',out);print('PASS M2',flush=True)
if __name__=='__main__':main()
