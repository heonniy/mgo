"""CPU-only structural, route/action and physical-slot audit of a PLAN artifact."""
import argparse,gzip,hashlib,json,pickle
from pathlib import Path
import numpy as np

def sha(x):return hashlib.sha256(x).hexdigest()
def validate(path):
 ranks=sorted(path.glob('rank[0-9].json'));world=len(ranks);assert world in (4,8)
 plans=[];out=[]
 for r,p in enumerate(ranks):
  receipt=json.loads(p.read_text());assert receipt['status']=='PASS' and receipt['phase']=='PLAN' and receipt['rank']==r
  source=path/f'rank{r}.pkl.gz'
  with gzip.open(source,'rb') as f:payload=f.read()
  assert sha(payload)==receipt['action_hash'],'serialized action payload changed'
  events=pickle.loads(payload);del payload
  assert len(events)==(receipt.get('decode_steps',256)+1)*48
  keys=np.full(receipt['cache_capacity'],-1,np.int32);route=hashlib.sha256();fetches=0;replicas=0
  for i,e in enumerate(events):
   assert e['layer']==i%48
   selected=np.asarray(e['selected']);assert selected.ndim==2 and selected.shape[1]==8 and np.all((selected>=0)&(selected<128))
   route.update(np.array(selected.shape,np.int64).tobytes());route.update(selected.tobytes())
   for key,slot,victim,rep in e['fetches']:
    assert key//128==i%48 and keys[slot]==victim and key not in keys
    keys[slot]=key;fetches+=1;replicas+=rep
   assert len(set(keys[keys>=0]))==np.count_nonzero(keys>=0)
   for expert,rows,cols,slot in e['groups']:
    assert keys[slot]==e['layer']*128+expert and len(rows)==len(cols) and len(rows)>0
   assert len(e['send_idx'])==sum(e['send_counts']) and len(e['return_order'])==sum(e['return_counts'])
  assert sha(keys.tobytes())==receipt['state_hash'];tokens=np.load(path/f'rank{r}_tokens.npy');assert sha(tokens.tobytes())==receipt['token_hash']
  out.append(dict(rank=r,status='PASS',route_hash=route.hexdigest(),action_hash=receipt['action_hash'],schedule_file_sha256=sha(source.read_bytes()),token_hash=receipt['token_hash'],state_hash=receipt['state_hash'],fetches=fetches,replica_fetches=replicas,physical_H2D_bytes=fetches*9437184,cache_slots=len(keys)));plans.append(events)
 assert len({len(x) for x in plans})==1
 for i in range(len(plans[0])):
  for src in range(world):
   for dst in range(world):
    assert plans[src][i]['send_counts'][dst]==plans[dst][i]['recv_counts'][src]
    assert plans[src][i]['return_counts'][dst]==plans[dst][i]['return_recv_counts'][src]
 result=dict(status='PASS',events=len(plans[0]),ranks=out)
 (path/'schedule_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status='PASS',path=str(path),world=world,events=len(plans[0]))),flush=True);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('path',type=Path);validate(p.parse_args().path)
