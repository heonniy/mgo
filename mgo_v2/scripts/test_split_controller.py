"""P=0 current-path parity and bounded CPU overhead with a warmed signature."""
import time,json,statistics
from pathlib import Path
import numpy as np
from env_offload_policy import Policy
from mgo_v2.controller import DecodePrefetchController
from mgo_v2.predictor import TransitionPredictor
root=Path('/home/hwlee/mgo-results/la_physical_validation_20261004/B128')
a={k:np.load(root/(k+'.npy'),mmap_mode='r') for k in ['selected','weights','offsets','prefill_origins','decode_origins','gates']}
seed=json.loads((root/'receipt.json').read_text())['winner']['placement_seed'];cap=[3686//8+(r<3686%8) for r in range(8)];pred=TransitionPredictor(np.full((47,128,128),1/128))
rows=[]
for policy,kind in [('BR',0),('CA',1),('LA',4)]:
 runs=[]
 for mode in ('legacy','split'):
  obj=Policy(cap,np.zeros((48,128,128),np.float32),False,kind,seed) if mode=='legacy' else DecodePrefetchController(cap,0,policy,seed,pred)
  times=[];actions=[]
  for event in range(432):
   lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];selected=np.asarray(a['selected'][lo:hi]);weights=np.asarray(a['weights'][lo:hi]);gate=a['gates'][event]
   start=time.perf_counter()
   if mode=='legacy':out=obj.apply(event,selected,weights,org,gate,np.zeros((128,8),np.int32))
   else:out,_,_=obj.plan_current(event,selected,weights,org,gate)
   elapsed=(time.perf_counter()-start)*1000
   if event>=96:times.append(elapsed)
   actions.append(out[5])
  main=obj if mode=='legacy' else obj.main
  runs.append((actions,main.slots.copy(),statistics.median(times)))
 assert runs[0][0]==runs[1][0] and np.array_equal(runs[0][1],runs[1][1])
 # 0.1ms absolute tolerance protects small CPU timings against host noise.
 assert runs[1][2]<=runs[0][2]*1.2+.1,(policy,runs[0][2],runs[1][2])
 rows.append(dict(policy=policy,legacy_median_ms=runs[0][2],split_median_ms=runs[1][2],actions_and_slots_equal=True))
p=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004/M7_cpu_controller_gate.json';p.write_text(json.dumps(dict(status='PASS',rows=rows,regression_tolerance='20% + 0.1ms absolute for microbench host noise'),indent=2)+'\n')
print(json.dumps(rows))
