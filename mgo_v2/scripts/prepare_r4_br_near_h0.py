"""CPU reference replay for the four existing R4 frozen decode32 workloads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import concurrent.futures,hashlib,json
from pathlib import Path
import numpy as np
from mgo_v2.controller import DecodePrefetchController
from mgo_v2.predictor import TransitionPredictor
ROOT=Path('/home/hwlee/mgo-results/r4_br_near_h0_20261007')
SOURCE=Path('/home/hwlee/mgo-results/strict_laca_headroom_20261006/physical')
TRANSITION=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/predictor/transition.npy')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def job(spec):
 batch,context=spec;label=f'B{batch}_L{context}';src=SOURCE/f'S2D32_R4_B{batch}_L{context}'/'frozen';dst=ROOT/label;dst.mkdir(parents=True,exist_ok=False)
 original=json.loads((src/'receipt.json').read_text());assert original['world']==4 and original['decode_steps']==32
 a={k:np.load(src/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
 proofs={}
 for policy in ['BR','LA_CA_NEAR']:
  caps=original['capacities'];c=DecodePrefetchController(caps,2,policy,original['placement_seed'],TransitionPredictor(np.load(TRANSITION)));copies=np.zeros(4,np.int64)
  for event in range(33*48):
   lo,hi=a['offsets'][event:event+2];sel=a['selected'][lo:hi];org=a['prefill_origins'] if event<48 else a['decode_origins']
   out,_,_=c.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
   for rank,*_ in out[5]:copies[rank]+=1
   if event>=48:
    hist=np.bincount((sel.astype(np.int64)+org.astype(np.int64)[:,None]*128).ravel(),minlength=4*128).reshape(4,128)
    for rank,*_ in c.plan_prefetch_next(hist):copies[rank]+=1
  c.arena.assert_consistent();assert not c.pending
  proofs[policy]=dict(status='PASS',horizon=32,P=2,policy=policy,rank_state_hashes=[digest(c.main.slots[r,:caps[r]]) for r in range(4)],rank_role_hashes=[digest(c.arena.main_physical[r]) for r in range(4)],counters=c.counters,max_copy_counts=copies.tolist())
 row=dict(label=label,batch=batch,context=context,inputs=str(src),proofs=proofs,receipt_sha256=sha(src/'receipt.json'),requests_sha256=sha(src/'requests.json'),teacher_sha256=sha(src/'teacher.npy'),predictor_sha256=sha(TRANSITION),placement_seed=original['placement_seed'],capacities=original['capacities'])
 (dst/'cpu_proofs.json').write_text(json.dumps(row,indent=2)+'\n');return row
if __name__=='__main__':
 ROOT.mkdir(exist_ok=False)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(job,[(16,256),(16,512),(64,256),(64,512)]))
 (ROOT/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS: four frozen workloads / eight P2 CPU proofs',flush=True)
