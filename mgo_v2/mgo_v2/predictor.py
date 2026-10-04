"""Immutable batch next-layer transition predictor; no future-route access."""
import numpy as np
from numba import njit

@njit(cache=True)
def transition_counts(routes):
 steps,layers,requests,topk=routes.shape
 counts=np.zeros((layers-1,128,128),np.int64)
 for step in range(steps):
  for layer in range(layers-1):
   for request in range(requests):
    for i in range(topk):
     for j in range(topk):counts[layer,routes[step,layer,request,i],routes[step,layer+1,request,j]]+=1
 return counts

def normalize(counts):
 counts=np.asarray(counts)
 if counts.ndim!=3 or np.any(counts<0):raise ValueError('invalid transition counts')
 sums=counts.sum(-1,keepdims=True)
 return np.divide(counts,sums,out=np.zeros(counts.shape,np.float64),where=sums!=0)

class TransitionPredictor:
 def __init__(self,table,topk=8):
  table=np.array(table,dtype=np.float64,copy=True)
  if table.ndim!=3 or table.shape[1:]!=(128,128) or not np.isfinite(table).all() or np.any(table<0):raise ValueError('invalid transition table')
  sums=table.sum(-1)
  if not np.all(np.isclose(sums,1)|np.isclose(sums,0)):raise ValueError('transition rows must normalize')
  table.flags.writeable=False;self.table=table;self.topk=topk
 def predict_next(self,layer,per_rank_counts):
  if layer<0 or layer>=len(self.table):return None
  h=np.asarray(per_rank_counts,dtype=np.float64)
  if h.ndim!=2 or h.shape[1]!=128 or np.any(h<0):raise ValueError('invalid histogram')
  return h@self.table[layer]/self.topk

def choose_candidates(predicted,main_keys,prefetch_keys,target_layer,budget,world=8):
 if predicted is None or budget==0:return np.empty(0,np.int64)
 unavailable=set(int(x) for x in main_keys if x>=0)|set(int(x) for x in prefetch_keys if x>=0)
 totals=predicted.sum(0);order=np.lexsort((np.arange(128),-totals))
 return np.array([int(e) for e in order if target_layer*128+int(e) not in unavailable and totals[e]>0][:world*budget],np.int64)
