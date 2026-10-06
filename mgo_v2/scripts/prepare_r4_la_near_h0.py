"""Replay only missing pure-LA CPU proof; reuse original Near and input hashes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
from prepare_r4_br_near_h0 import digest,sha,TRANSITION
from mgo_v2.controller import DecodePrefetchController
from mgo_v2.predictor import TransitionPredictor
ROOT=Path('/home/hwlee/mgo-results/r4_la_near_h0_20261007')
if __name__=='__main__':
 assert not ROOT.exists();ROOT.mkdir()
 source=Path('/home/hwlee/mgo-results/r4_br_near_h0_20261007/manifest.json')
 spec=next(x for x in json.loads(source.read_text()) if x['label']=='B64_L512');src=Path(spec['inputs']);caps=spec['capacities']
 for file,key in [('receipt.json','receipt_sha256'),('requests.json','requests_sha256'),('teacher.npy','teacher_sha256')]:assert sha(src/file)==spec[key]
 assert sha(TRANSITION)==spec['predictor_sha256']
 a={k:np.load(src/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
 c=DecodePrefetchController(caps,2,'LA',spec['placement_seed'],TransitionPredictor(np.load(TRANSITION)));copies=np.zeros(4,np.int64)
 for event in range(33*48):
  lo,hi=a['offsets'][event:event+2];sel=a['selected'][lo:hi];org=a['prefill_origins'] if event<48 else a['decode_origins']
  out,_,_=c.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
  for rank,*_ in out[5]:copies[rank]+=1
  if event>=48:
   hist=np.bincount((sel.astype(np.int64)+org.astype(np.int64)[:,None]*128).ravel(),minlength=512).reshape(4,128)
   for rank,*_ in c.plan_prefetch_next(hist):copies[rank]+=1
 c.arena.assert_consistent();assert not c.pending
 proof=dict(status='PASS',horizon=32,P=2,policy='LA',rank_state_hashes=[digest(c.main.slots[r,:caps[r]]) for r in range(4)],rank_role_hashes=[digest(c.arena.main_physical[r]) for r in range(4)],counters=c.counters,max_copy_counts=copies.tolist())
 spec['proofs']={'LA':proof,'LA_CA_NEAR':spec['proofs']['LA_CA_NEAR']}
 (ROOT/spec['label']).mkdir();(ROOT/'manifest.json').write_text(json.dumps([spec],indent=2)+'\n')
 print('PASS: pure LA CPU reference, original Near proof and input hashes')
