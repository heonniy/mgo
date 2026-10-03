"""Exact layout/order parity against the explicit physical routing definition."""
import ast,json,time
from pathlib import Path
import numpy as np
from env_offload_layout import plan_layout
root=Path(__file__).resolve().parents[1]
# The explicit reference is preserved from the already launched first PLAN.
source=Path('/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003/P_BR_env1_PLAN_3/worker_source.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='plan_layout');scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<explicit layout>','exec'),scope);reference=scope['plan_layout']
rng=np.random.default_rng(42);cases=0;old_time=0;new_time=0
for world in [4,8]:
 for count in [0,1,16,64,512]:
  counts=[count+(r%2) for r in range(world)];origins=np.repeat(np.arange(world),counts);n=len(origins)
  lengths=rng.integers(1,9,n);effective=np.stack([rng.choice(128,8,replace=False) for _ in range(n)])
  dest=rng.integers(0,world,(n,8))
  for rank in range(world):
   start=time.monotonic();a=reference(effective,lengths,dest,origins,counts,rank);old_time+=time.monotonic()-start
   start=time.monotonic();b=plan_layout(effective,lengths,dest,origins,counts,rank);new_time+=time.monotonic()-start
   assert a==b,(world,count,rank,[k for k in a if a[k]!=b[k]]);cases+=1
result=dict(status='PASS',cases=cases,reference_seconds=old_time,vectorized_seconds=new_time,speedup=old_time/new_time)
(root/'experiments/env_e2e_tpot_offload_20261003/layout_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
