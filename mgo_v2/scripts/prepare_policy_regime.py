"""Owner R4/decode64 pivot: reuse exact inputs, no GPU or trace recapture."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import json,hashlib,concurrent.futures
from pathlib import Path
import numpy as np
import psutil
from mgo_v2.controller import DecodePrefetchController
from mgo_v2.predictor import TransitionPredictor
BASE=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
ROOT=Path('/home/hwlee/mgo-results/policy_regime_20261005')
FROZEN=BASE/'R4_H64/inputs_B128_H64'
POLICIES=('BR','OLD_CA','FCA','LA_CA')
SOURCE=Path('/home/hwlee/mgo-results/la_physical_validation_20261004')
PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def proof(spec):
 setting,policy,budget=spec;batch=128;dst=ROOT/setting/'inputs_B128_H64';target=dst/f'{policy}_P{budget}_proof.json'
 if target.exists():return json.loads(target.read_text())
 assert psutil.virtual_memory().available>256*2**30
 meta=json.loads((dst/'input_receipt.json').read_text());a={k:np.load(dst/f'{k}.npy',mmap_mode='r') for k in ('selected','weights','offsets','prefill_origins','decode_origins','gates')}
 caps=[461]*4 if setting=='C30' else [922,922,921,921];c=DecodePrefetchController(caps,budget,policy,meta['placement_seed'],TransitionPredictor(np.load(BASE/'predictor/transition.npy')));copies=np.zeros(4,np.int64);fetches=[]
 for event in range(65*48):
  lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];sel=a['selected'][lo:hi]
  out,_,_=c.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
  for rank,*_ in out[5]:copies[rank]+=1
  if budget==0:fetches.append(out[5])
  if budget and event>=48:
   hist=np.bincount((sel.astype(np.int64)+org.astype(np.int64)[:,None]*128).ravel(),minlength=512).reshape(4,128)
   for rank,*_ in c.plan_prefetch_next(hist):copies[rank]+=1
 c.arena.assert_consistent();assert not c.pending
 digest=lambda x:hashlib.sha256(x.tobytes()).hexdigest()
 d=dict(status='PASS',world=4,batch=batch,policy=policy,P=budget,horizon=64,rank_state_hashes=[digest(c.main.slots[r,:caps[r]]) for r in range(4)],rank_role_hashes=[digest(c.arena.main_physical[r]) for r in range(4)],counters=c.counters,max_copy_counts=copies.tolist())
 if budget==0:
  f=dst/f'{policy}_fetches.json';write(f,fetches);d.update(fetches_sha256=sha(f),H2D_bytes=(copies*9437184).tolist())
 write(target,d);return {k:d[k] for k in ('status','batch','policy','P')}
def main():
 ROOT.mkdir(exist_ok=True)
 for setting in ('C30','C60'):
  dst=ROOT/setting/'inputs_B128_H64';dst.mkdir(parents=True,exist_ok=True)
  for name in ('selected.npy','weights.npy','offsets.npy','prefill_origins.npy','decode_origins.npy','gates.npy','teacher.npy','requests.json','input_receipt.json'):
   if not (dst/name).exists():(dst/name).symlink_to(FROZEN/name)
 specs=[(c,p,k) for c in ('C30','C60') for p in POLICIES for k in (0,2)]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for result in pool.map(proof,specs):print(result,flush=True)
 for setting in ('C30','C60'):
  dst=ROOT/setting/'inputs_B128_H64';meta=json.loads((dst/'input_receipt.json').read_text())
  write(dst/'receipt.json',dict(status='PASS',world=4,batch=128,horizon=64,capacities=[461]*4 if setting=='C30' else [922,922,921,921],winner={'placement_seed':meta['placement_seed']},proofs={p:json.loads((dst/f'{p}_P0_proof.json').read_text()) for p in POLICIES},source_receipt_sha256=sha(dst/'input_receipt.json')))
 write(PACKET/'POLICY_REGIME_CPU_PROOFS.json',dict(status='PASS',world=4,horizon=64,batch=128,policies=POLICIES,proofs={str(p):sha(p) for p in ROOT.glob('*/inputs*/*_proof.json')}))
if __name__=='__main__':main()
