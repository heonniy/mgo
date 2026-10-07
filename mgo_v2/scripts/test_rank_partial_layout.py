"""Independent legacy-layout parity including empty/uneven/full-size cases."""
import time,json
import numpy as np
import torch
from env_offload_layout import plan_layout,plan_rank_partial_layout
from env_offload_tensors import pack_layouts,pack_rank_partial_layout

def main():
 rng=np.random.default_rng(143);checks=0;timings=[]
 for counts,kind in [([0]*4,'empty'),([1,0,3,2],'uneven'),([1]*4,'none'),([16]*4,'mixed'),([8]*4,'single-owner'),([4096]*4,'prefill-small'),([32768]*4,'prefill-large')]:
  n=sum(counts);origins=np.repeat(np.arange(4),counts)
  effective=np.argsort(rng.random((n,128)),axis=1)[:,:8].astype(np.int32)
  lengths=np.zeros(n,np.int32) if kind=='none' else (rng.integers(0,9,n,dtype=np.int32) if kind=='mixed' else np.full(n,8,np.int32))
  owners=np.zeros(128,dtype=np.int64) if kind=='single-owner' else rng.integers(0,4,128);destinations=owners[effective]
  for rank in range(4):
   t=time.perf_counter();old=plan_layout(effective,lengths,destinations,origins,counts,rank);oldtime=time.perf_counter()-t
   plan_rank_partial_layout(effective,lengths,destinations,origins,counts,rank) # exclude JIT compilation
   t=time.perf_counter();new=plan_rank_partial_layout(effective,lengths,destinations,origins,counts,rank);newtime=time.perf_counter()-t
   assert old['send_counts']==new['send_counts'] and old['recv_counts']==new['recv_counts']
   for key in ('send_idx','send_eids'):np.testing.assert_array_equal(np.asarray(old[key]).reshape(np.asarray(new[key]).shape),new[key])
   assert len(old['groups'])==len(new['groups'])
   for a,b in zip(old['groups'],new['groups']):
    assert a[0]==b[0];np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
   for e in (old,new):
    e['targets']=np.arange(128);e['groups']=[(*x,x[0]) for x in e['groups']]
   old['selected']=effective[origins==rank]
   t=time.perf_counter();po=pack_layouts([old],device='cpu')[0];oldpack=time.perf_counter()-t
   t=time.perf_counter();pn=pack_rank_partial_layout(new,device='cpu');newpack=time.perf_counter()-t
   for key in ('send_idx','send_eids','targets'):assert torch.equal(po[key],pn[key]),key
   for a,b in zip(po['groups'],pn['groups']):assert a[0]==b[0] and a[3]==b[3] and torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
   assert 'return_order' not in pn and 'combine' not in pn and 'selected' not in pn
   checks+=1
   if rank==0:timings.append(dict(case=kind,old_layout_seconds=oldtime,new_layout_seconds=newtime,old_pack_seconds=oldpack,new_pack_seconds=newpack))
 print(json.dumps(dict(status='PASS',cases=checks,cpu_only_timings=timings),indent=2))
if __name__=='__main__':main()
