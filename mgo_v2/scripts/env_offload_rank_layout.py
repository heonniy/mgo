"""Linear packet construction in canonical peer/token/expert order."""
import numpy as np
from numba import njit

@njit(cache=True,nogil=True)
def build_rank_partial(effective,lengths,destinations,counts,rank,experts=128):
 world=len(counts);offsets=np.empty(world+1,np.int64);offsets[0]=0
 for r in range(world):offsets[r+1]=offsets[r]+counts[r]
 n=offsets[-1];local=counts[rank]
 topk=effective.shape[1]
 send_idx=np.empty(local*world,np.int64)
 send_ids=np.full((local*world,topk),-1,np.int64)
 send_counts=np.zeros(world,np.int64);sent=0
 scratch=np.empty(topk,np.int64)
 for peer in range(world):
  for token in range(offsets[rank],offsets[rank+1]):
   size=0
   for j in range(lengths[token]):
    if destinations[token,j]==peer:
     expert=effective[token,j];pos=size
     while pos>0 and scratch[pos-1]>expert:
      scratch[pos]=scratch[pos-1];pos-=1
     scratch[pos]=expert;size+=1
   if size:
    send_idx[sent]=token-offsets[rank]
    for j in range(size):send_ids[sent,j]=scratch[j]
    sent+=1;send_counts[peer]+=1
 recv_ids=np.full((n,topk),-1,np.int64);recv_counts=np.zeros(world,np.int64)
 received=0;expert_counts=np.zeros(experts,np.int64)
 for peer in range(world):
  for token in range(offsets[peer],offsets[peer+1]):
   size=0
   for j in range(lengths[token]):
    if destinations[token,j]==rank:
     expert=effective[token,j];pos=size
     while pos>0 and scratch[pos-1]>expert:
      scratch[pos]=scratch[pos-1];pos-=1
     scratch[pos]=expert;size+=1
   if size:
    for j in range(size):
     expert=scratch[j];recv_ids[received,j]=expert;expert_counts[expert]+=1
    received+=1;recv_counts[peer]+=1
 starts=np.empty(experts+1,np.int64);starts[0]=0
 for expert in range(experts):starts[expert+1]=starts[expert]+expert_counts[expert]
 rows=np.empty(starts[-1],np.int64);cols=np.empty(starts[-1],np.int64);cursor=starts[:-1].copy()
 for row in range(received):
  for col in range(topk):
   expert=recv_ids[row,col]
   if expert<0:break
   pos=cursor[expert];rows[pos]=row;cols[pos]=col;cursor[expert]+=1
 return send_idx[:sent],send_ids[:sent],send_counts,recv_counts,starts,rows,cols

@njit(cache=True,nogil=True)
def layer_physical_slots(slots,physical,layer,experts=128):
 """One scan of current MAIN roles; rebuilt after promotions/evictions."""
 lookup=np.full(experts,-1,np.int64)
 for slot in range(len(physical)):
  key=slots[slot]
  if key>=0 and key//experts==layer:
   expert=key%experts
   if lookup[expert]!=-1:raise ValueError('duplicate main expert')
   lookup[expert]=physical[slot]
 return lookup
