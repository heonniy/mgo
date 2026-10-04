"""Freeze short BR/CA/LA references without rerouting or recapturing the model."""
import json,hashlib
from pathlib import Path
import numpy as np
from env_offload_policy import Policy
from r8_phase_aware_policy_cpu import replay_arrays,as_result_dict,capacities
from ca_stress_common import write,sha
P=Path(__file__).resolve().parents[1];PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')

def main():
 src=Path('/home/hwlee/mgo-results/la_physical_validation_20261004/B128');dst=ROOT/'baseline_B128';dst.mkdir(parents=True,exist_ok=True)
 old=json.loads((src/'receipt.json').read_text());a={k:np.load(src/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
 for f in list(src.glob('*.npy'))+[src/'requests.json']:
  target=dst/f.name
  if not target.exists():target.symlink_to(f)
 proofs={}
 for name,kind in [('BR',0),('CA',1),('LA',4)]:
  oracle=as_result_dict(replay_arrays(a,128,old['winner']['placement_seed'],2 if kind==4 else kind,False,horizon=8))
  pol=Policy(capacities(),np.zeros((48,128,128),np.float32),False,kind,old['winner']['placement_seed']);fetches=[]
  for event in range(432):
   lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins']
   _,_,_,_,dsts,fs,_=pol.apply(event,a['selected'][lo:hi],a['weights'][lo:hi],org,a['gates'][event],np.zeros((128,8),np.int32))
   assert np.array_equal(np.bincount([f[0] for f in fs],minlength=8),oracle['h2d_counts'][event])
   assert np.bincount(dsts.ravel(),minlength=8).max()==oracle['max_rows'][event]
   fetches.append(fs)
  for key in ['slots','owner','primary']:assert np.array_equal(getattr(pol,key),oracle[key])
  write(dst/(name+'_fetches.json'),fetches)
  proofs[name]=dict(rank_state_hashes=[hashlib.sha256(pol.slots[r,:capacities()[r]].tobytes()).hexdigest() for r in range(8)],H2D_bytes=(oracle['h2d_counts'].sum(0)*9437184).tolist(),fetches_sha256=sha(dst/(name+'_fetches.json')),status='PASS',events=432)
 write(dst/'receipt.json',dict(status='PASS',horizon=8,winner=old['winner'],proofs=proofs))
 write(PACKET/'M0_cpu_parity.json',dict(status='PASS',policies=list(proofs),events=432,reference_path=str(dst),source_receipt_sha256=sha(src/'receipt.json'),proofs=proofs))
 print('PASS M0 BR/CA/LA oracle adapter parity')
if __name__=='__main__':main()
