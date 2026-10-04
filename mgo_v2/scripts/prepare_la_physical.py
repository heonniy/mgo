"""Freeze LOAD winners and prove the live adapter against the complete CPU replay."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from r8_phase_aware_policy_cpu import Pool,make_arrays,replay_arrays,as_result_dict,capacities,state_hash
from env_offload_policy import Policy
from ca_stress_common import write,sha
P=Path(__file__).resolve().parents[1]
SOURCE=P/'experiments/r8_real_replica_batch_scaling_20261004'
PACKET=P/'experiments/la_physical_validation_20261004'
ROOT=Path('/home/hwlee/mgo-results/la_physical_validation_20261004')

def main():
 PACKET.mkdir(exist_ok=True);ROOT.mkdir(exist_ok=True)
 winners=json.loads((SOURCE/'stress_winners.json').read_text());previous=json.loads((SOURCE/'RESULT.json').read_text())
 for batch in (128,256):
  dest=ROOT/f'B{batch}';dest.mkdir(exist_ok=True)
  if (dest/'receipt.json').exists():continue
  win=winners[f'B{batch}_LOAD'];pool=Pool(win['dataset']);order,a=make_arrays(pool,batch,win['sample_seed'],win['dp_seed'])
  for key,value in a.items():np.save(dest/(key+'.npy'),value)
  reqpath=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004')/(win['dataset']+'_requests.json')
  requests=json.loads(reqpath.read_text())['requests']
  write(dest/'requests.json',dict(ranks=[[requests[int(i)] for i in order[r*batch:(r+1)*batch]] for r in range(8)],request_ids=order.tolist(),source_sha256=sha(reqpath)))
  np.save(dest/'teacher.npy',np.load(pool.path/'generated_tokens.npy',mmap_mode='r')[order,:256])
  proofs={}
  for name,oracle_index,adapter_index in [('BR',0,0),('LA',2,4)]:
   ref=replay_arrays(a,batch,win['placement_seed'],oracle_index,False,horizon=256);rd=as_result_dict(ref)
   old=next(x for x in previous['selected_full'][f'B{batch}_LOAD'] if x['policy']==name)
   assert state_hash(ref)==old['state_sha256']
   pol=Policy(capacities(),np.zeros((48,128,128),np.float32),False,adapter_index,win['placement_seed'])
   events=[]
   for event in range(257*48):
    lo,hi=a['offsets'][event:event+2];orig=a['prefill_origins'] if event<48 else a['decode_origins']
    _,effective,_,lengths,dsts,fetches,row=pol.apply(event,a['selected'][lo:hi],a['weights'][lo:hi],orig,a['gates'][event],np.zeros((128,8),np.int32))
    loads=np.bincount(dsts[dsts>=0],minlength=8);counts=np.bincount([f[0] for f in fetches],minlength=8)
    assert loads.max()==rd['max_rows'][event],(name,event,'loads')
    assert np.array_equal(counts,rd['h2d_counts'][event]),(name,event,'fetches')
    assert np.count_nonzero(dsts!=orig[:,None])==rd['peer_return_routes'][event]
    events.append(fetches)
    if event%768==0:print(json.dumps(dict(batch=batch,policy=name,event=event,total=12336)),flush=True)
   for key in ('slots','owner','primary'):assert np.array_equal(getattr(pol,key),rd[key]),key
   write(dest/(name+'_fetches.json'),events)
   proofs[name]=dict(status='PASS',cpu_state_sha256=state_hash(ref),rank_state_hashes=[hashlib.sha256(pol.slots[r,:capacities()[r]].tobytes()).hexdigest() for r in range(8)],H2D_bytes=rd['h2d_counts'].sum(0).astype(np.int64).__mul__(9437184).tolist(),critical_rows_sum=int(rd['max_rows'][48:].sum()),event_exact_checks=12336,fetches_sha256=sha(dest/(name+'_fetches.json')))
  receipt=dict(status='PASS',batch=batch,world=8,dataset=win['dataset'],winner=win,proofs=proofs,files={f.name:sha(f) for f in dest.glob('*.npy')},request_sha256=sha(dest/'requests.json'))
  write(dest/'receipt.json',receipt);write(PACKET/f'B{batch}_adapter_validation.json',receipt)
  print('VALIDATED',batch,flush=True)
if __name__=='__main__':main()
