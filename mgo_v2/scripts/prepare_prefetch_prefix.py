"""CPU logical-state and charged-copy bounds for physical prefetch prefix gates."""
import json,hashlib
from pathlib import Path
import numpy as np
from mgo_v2.controller import DecodePrefetchController
from mgo_v2.predictor import TransitionPredictor
from calibrate_refactor_predictor import histogram
from ca_stress_common import write
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
def main():
 dst=ROOT/'baseline_B128';source=json.loads((dst/'receipt.json').read_text());a={k:np.load(dst/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']};caps=[3686//8+(r<3686%8) for r in range(8)]
 for name in ['BR','CA','LA']:
  c=DecodePrefetchController(caps,2,name,source['winner']['placement_seed'],TransitionPredictor(np.load(ROOT/'predictor/transition.npy')));copies=np.zeros(8,np.int64)
  for event in range(432):
   lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];sel=a['selected'][lo:hi]
   out,_,_=c.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
   for rank,*_ in out[5]:copies[rank]+=1
   for rank,*_ in c.plan_prefetch_next(histogram(sel,org)):copies[rank]+=1
   c.arena.assert_consistent()
  write(dst/f'{name}_P2_proof.json',dict(status='PASS',rank_state_hashes=[hashlib.sha256(c.main.slots[r,:caps[r]].tobytes()).hexdigest() for r in range(8)],counters=c.counters,max_copy_counts=copies.tolist()))
  print(name,c.counters,flush=True)
if __name__=='__main__':main()
