"""Derive decode64 expected tokens/cache states from the accepted full PLAN."""
import gzip,hashlib,json,pickle
from pathlib import Path
import numpy as np
ROOT=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004')
PLAN=Path('/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003/P_CA-rep_env1_PLAN_0')
PACKET=Path(__file__).resolve().parents[1]/'experiments/timing_stability_numa_20261004'
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
 validation=json.loads((PLAN/'schedule_validation.json').read_text());assert validation['status']=='PASS'
 rows=[]
 for rank in range(8):
  p=PLAN/f'rank{rank}.pkl.gz';assert sha(p.read_bytes())==validation['ranks'][rank]['schedule_file_sha256']
  with gzip.open(p,'rb') as f:events=pickle.load(f)
  receipt=json.loads((PLAN/f'rank{rank}.json').read_text());keys=np.full(receipt['cache_capacity'],-1,np.int32);route=hashlib.sha256()
  for e in events[:65*48]:
   for key,slot,victim,rep in e['fetches']:assert keys[slot]==victim;keys[slot]=key
   selected=e['selected'];route.update(np.array(selected.shape,np.int64).tobytes());route.update(selected.tobytes())
  tokens=np.load(PLAN/f'rank{rank}_tokens.npy');assert sha(tokens.tobytes())==receipt['token_hash']
  rows.append(dict(rank=rank,events=65*48,token_hash=sha(np.ascontiguousarray(tokens[:,:64]).tobytes()),state_hash=sha(keys.tobytes()),route_hash=route.hexdigest(),schedule_file_sha256=validation['ranks'][rank]['schedule_file_sha256']))
  del events
 out=dict(status='PASS',horizon=64,source_plan=str(PLAN),convention='prefill plus first 64 decode forwards; g1..g64 output prefix; discard last logits',ranks=rows)
 for p in [ROOT/'prefix_validation.json',PACKET/'prefix_validation.json']:p.write_text(json.dumps(out,indent=2)+'\n')
 print('PASS exact frozen decode64 prefix, all eight ranks')
if __name__=='__main__':main()
