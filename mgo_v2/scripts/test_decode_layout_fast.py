"""CPU differential tests: decode packet order, compact tensors, role lookup."""
import json
import numpy as np
import torch
from env_offload_layout import plan_layout,plan_rank_partial_layout
from env_offload_tensors import pack_layouts,pack_rank_partial_layout
from env_offload_rank_layout import layer_physical_slots

def main():
 rng=np.random.default_rng(708);checks=0
 for counts in ([0,0,0,0],[0,1,2,3],[16]*4,[32]*4,[64]*4):
  n=sum(counts);origins=np.repeat(np.arange(4),counts)
  for trial in range(5):
   eff=np.argsort(rng.random((n,128)),axis=1)[:,:8].astype(np.int32)
   lengths=np.full(n,8,np.int32) if trial==0 else rng.integers(0,9,n,dtype=np.int32)
   owners=rng.integers(0,4,128);dst=owners[eff]
   for rank in range(4):
    old=plan_layout(eff,lengths,dst,origins,counts,rank);new=plan_rank_partial_layout(eff,lengths,dst,origins,counts,rank)
    keys=np.arange(128,dtype=np.int64)+128*trial;slots=np.full(460,-1,np.int64);ix=rng.choice(460,128,replace=False);slots[ix]=keys
    physical=rng.permutation(462)[:460];lookup=layer_physical_slots(slots,physical,trial)
    for e in range(128):assert lookup[e]==physical[np.flatnonzero(slots==trial*128+e)[0]]
    for x in (old,new):x['targets']=np.arange(128);x['groups']=[(*g,int(lookup[g[0]])) for g in x['groups']]
    old['selected']=eff[origins==rank];a=pack_layouts([old],device='cpu')[0];b=pack_rank_partial_layout(new,device='cpu')
    assert a['send_counts']==b['send_counts'] and a['recv_counts']==b['recv_counts']
    for k in ('send_idx','send_eids','targets'):assert torch.equal(a[k],b[k]),k
    assert len(a['groups'])==len(b['groups'])
    for ga,gb in zip(a['groups'],b['groups']):assert ga[0]==gb[0] and ga[3]==gb[3] and torch.equal(ga[1],gb[1]) and torch.equal(ga[2],gb[2])
    # Simulate a promotion role swap, then eviction: lookup must not be cached.
    physical[0],physical[1]=physical[1],physical[0];slots[ix[0]]=-1
    after=layer_physical_slots(slots,physical,trial);assert after[0]==-1
    for e in range(1,128):assert after[e]==physical[np.flatnonzero(slots==trial*128+e)[0]]
    checks+=1
 print(json.dumps(dict(status='PASS',differential_cases=checks,checks='packet counts/indices/group order/physical slots; promotion/eviction refresh; empty and uneven ranks')))
if __name__=='__main__':main()
