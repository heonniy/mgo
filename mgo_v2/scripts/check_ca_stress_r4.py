"""CPU compatibility gate: selected full256 input, incremental physical PLAN policy."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib
from pathlib import Path
import numpy as np
from ca_stress_cpu import Pool,exact
from env_offload_policy import Policy
from ca_stress_common import digest
from run_ca_stress_r4 import PACKET,write
pool=Pool('ShareGPT');order=pool.candidate(4,8,205,4)
m=json.loads((PACKET.parent.parent/'ca_stress_workload_search_20261004/selected_manifests/ShareGPT_R4_B8_c30.json').read_text());assert order.tolist()==m['request_ids']
a=pool.pack(order,4,8);slots=1843;cap=np.array([slots//4+(r<slots%4) for r in range(4)]);sim=np.zeros((48,128,128),np.float32)
old=json.loads(Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/candidates/ShareGPT_R4_B8_s205_d4.json').read_text());cell=next(c for c in old['results'] if c['cache']==30)
assert hashlib.sha256(a['selected'].tobytes()).hexdigest()==old['route_sha256'] and hashlib.sha256(a['gates'].tobytes()).hexdigest()==old['gate_sha256']
checks=[]
for name,index in [('BR',0),('CA',1)]:
 ref=exact(a,4,30,name,73 if name=='BR' else 42);expected=cell['CA'] if name=='CA' else next(r for r in cell['BR'] if r['seed']==73);assert ref==expected
 policy=Policy(cap,sim,False,index,73);rows=[]
 for event in range(257*48):
  lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];rows.append(policy.apply(event,a['selected'][lo:hi],a['weights'][lo:hi],org,a['gates'][event],np.zeros((128,4),np.int32))[-1])
 h=hashlib.sha256()
 for v in [policy.slots,policy.owner,policy.primary,policy.last,policy.seen,policy.lost,policy.birth,policy.reuses,np.array([0,0],np.int64)]:h.update(v.tobytes())
 assert h.hexdigest()==expected['final_state_sha256'];rows=np.array(rows);peer=int((rows[:,29].sum()+rows[:,30].sum())*4096);assert peer==expected['full']['peer_bytes']
 checks.append(dict(policy=name,peer_bytes=peer,final_state_sha256=h.hexdigest(),quota_max_minus_min=int((rows[:,44]-rows[:,45]).max())))
write(PACKET/'implementation_validation.json',dict(status='PASS',full256_incremental_policy_equals_frozen_CPU=True,selected_manifest_preserved=True,rows=checks));print(checks,flush=True)
