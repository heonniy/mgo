"""Independent CPU proofs for frozen-trace physical timing configurations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,argparse,concurrent.futures
from pathlib import Path
import numpy as np
import psutil
from mgo_v2.controller import DecodePrefetchController
from mgo_v2.predictor import TransitionPredictor
from calibrate_refactor_predictor import histogram
from ca_stress_common import write,sha
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004');SOURCE=Path('/home/hwlee/mgo-results/la_physical_validation_20261004')
def digest(x):return hashlib.sha256(x.tobytes()).hexdigest()
def job(spec):
 batch,policy,budget,horizon=spec;dst=ROOT/f'inputs_B{batch}_H{horizon}';output=dst/f'{policy}_P{budget}_proof.json'
 if output.exists():return json.loads(output.read_text())
 assert psutil.virtual_memory().available>256*2**30
 src=SOURCE/f'B{batch}';original=json.loads((src/'receipt.json').read_text());a={k:np.load(src/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']};caps=[3686//8+(r<3686%8) for r in range(8)]
 c=DecodePrefetchController(caps,budget,policy,original['winner']['placement_seed'],TransitionPredictor(np.load(ROOT/'predictor/transition.npy')));copies=np.zeros(8,np.int64);fetches=[]
 for event in range((horizon+1)*48):
  lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];sel=a['selected'][lo:hi]
  out,_,_=c.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
  for rank,*_ in out[5]:copies[rank]+=1
  if budget==0:fetches.append(out[5])
  if budget and event>=48:
   for rank,*_ in c.plan_prefetch_next(histogram(sel,org)):copies[rank]+=1
 c.arena.assert_consistent();assert not c.pending
 proof=dict(status='PASS',batch=batch,policy=policy,P=budget,horizon=horizon,rank_state_hashes=[digest(c.main.slots[r,:caps[r]]) for r in range(8)],rank_role_hashes=[digest(c.arena.main_physical[r]) for r in range(8)],counters=c.counters,max_copy_counts=copies.tolist())
 if budget==0:
  path=dst/(policy+'_fetches.json');write(path,fetches);proof.update(fetches_sha256=sha(path),H2D_bytes=(copies*9437184).tolist())
  if horizon==256 and policy in original['proofs']:
   old=original['proofs'][policy];assert proof['rank_state_hashes']==old['rank_state_hashes'] and proof['H2D_bytes']==old['H2D_bytes']
 write(output,proof);return proof

def main(a):
 specs=[]
 for batch in a.batches:
  src=SOURCE/f'B{batch}';dst=ROOT/f'inputs_B{batch}_H{a.horizon}';dst.mkdir(exist_ok=True)
  for f in list(src.glob('*.npy'))+[src/'requests.json']:
   if not (dst/f.name).exists():(dst/f.name).symlink_to(f)
  for policy in a.policies:
   for p in sorted(set([0]+a.budgets)):specs.append((batch,policy,p,a.horizon))
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for row in pool.map(job,specs):print({k:row[k] for k in ['batch','policy','P','horizon','status']},flush=True)
 for batch in a.batches:
  src=SOURCE/f'B{batch}';dst=ROOT/f'inputs_B{batch}_H{a.horizon}';old=json.loads((src/'receipt.json').read_text())
  proofs={p:json.loads((dst/f'{p}_P0_proof.json').read_text()) for p in a.policies};write(dst/'receipt.json',dict(status='PASS',horizon=a.horizon,winner=old['winner'],proofs=proofs,source_receipt_sha256=sha(src/'receipt.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',type=int,default=[128,256]);p.add_argument('--policies',nargs='+',default=['BR']);p.add_argument('--budgets',nargs='+',type=int,default=[1,2,4]);p.add_argument('--horizon',type=int,default=64);main(p.parse_args())
