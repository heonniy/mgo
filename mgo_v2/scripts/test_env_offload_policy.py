"""Compare incremental PLAN policy against frozen full CPU replay on real input."""
import json
from pathlib import Path
import numpy as np
from br_carep_cpu import replay,suffix_demand
from env_offload_policy import Policy
R=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
P=Path(__file__).resolve().parents[1]/'experiments/br_ca_carep_cpu_headroom_20261003'
sim=np.load(json.loads((P/'similarity_audit.json').read_text())['similarity_path'])
pack=R/'packed/ShareGPT/R8_B16';a={k:np.load(pack/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
cap=np.array([2457//8+(r<2457%8) for r in range(8)])
results=[]
for sub,policy in [(False,0),(False,1),(True,0),(True,1),(True,2)]:
 horizon=8;ref=replay(a['selected'],a['weights'],a['offsets'],a['prefill_origins'],a['decode_origins'],a['gates'],sim,cap,horizon,True,sub,policy,42)
 future=suffix_demand(a['selected'],a['offsets'],a['decode_origins'],48,128,8,horizon)
 p=Policy(cap,sim,sub,policy);rows=[]
 for event in range((horizon+1)*48):
  lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins']
  result=p.apply(event,a['selected'][lo:hi],a['weights'][lo:hi],org,a['gates'][event],future[event//48,event%48]);rows.append(result[-1])
 assert np.array_equal(rows,ref[0]),(sub,policy,'metrics')
 for value,expected in zip([p.slots,p.owner,p.primary,p.last,p.seen,p.lost,p.birth,p.reuses],ref[2:10]):assert np.array_equal(value,expected),(sub,policy,'state')
 results.append(dict(substitution=sub,policy=policy,events=len(rows),status='PASS'));print(results[-1],flush=True)
out=P.parent/'env_e2e_tpot_offload_20261003/policy_validation.json';out.write_text(json.dumps(results,indent=2)+'\n')
