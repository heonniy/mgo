"""Check actual replica admissions against archived 256-decode CPU rows."""
import json
from pathlib import Path
import numpy as np
from br_carep_cpu import suffix_demand
from env_offload_policy import Policy
R=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003');P=Path(__file__).resolve().parents[1]/'experiments'
sim=np.load(json.loads((P/'br_ca_carep_cpu_headroom_20261003/similarity_audit.json').read_text())['similarity_path']);out=[]
for dataset,world,batch,cache in [('ShareGPT',8,16,40),('MATH',4,64,30)]:
 pack=R/'packed'/dataset/f'R{world}_B{batch}';a={k:np.load(pack/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
 ref=np.load(R/'events'/f'{dataset}_h256_R{world}_B{batch}_c{cache}_gate_s1_CA-rep_seed42.npz')['rows'];total=48*128*cache//100;cap=np.array([total//world+(r<total%world) for r in range(world)])
 future=suffix_demand(a['selected'],a['offsets'],a['decode_origins'],48,128,world,256);p=Policy(cap,sim,True,2);admissions=0
 for event in range(257*48):
  lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];result=p.apply(event,a['selected'][lo:hi],a['weights'][lo:hi],org,a['gates'][event],future[event//48,event%48]);assert np.array_equal(result[-1],ref[event]),(dataset,event)
  admissions+=sum(bool(f[-1]) for f in result[-2])
 assert admissions>0
 out.append(dict(dataset=dataset,world=world,batch=batch,events=12336,actual_replica_admissions=admissions,status='PASS'));print(out[-1],flush=True)
(P/'env_e2e_tpot_offload_20261003/replica_policy_validation.json').write_text(json.dumps(out,indent=2)+'\n')
