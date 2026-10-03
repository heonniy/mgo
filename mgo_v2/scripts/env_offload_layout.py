"""Vectorized immutable communication indices; PLAN only, no Torch imports."""
import numpy as np

def plan_layout(effective,lengths,destinations,origins,counts,rank):
 world=len(counts);offsets=np.cumsum([0]+counts);token,slot=np.nonzero(np.arange(effective.shape[1])[None,:]<lengths[:,None])
 experts=effective[token,slot].astype(np.int64);src=origins[token].astype(np.int64);dst=destinations[token,slot].astype(np.int64);local=token-offsets[src]
 def packets(keep,peers):
  e=experts[keep];t=local[keep];p=peers[keep];order=np.lexsort((e,t,p));e=e[order];t=t[order];p=p[order]
  if not len(e):return np.empty(0,np.int64),np.empty((0,8),np.int64),np.empty(0,np.int64)
  starts=np.flatnonzero(np.r_[True,(p[1:]!=p[:-1])|(t[1:]!=t[:-1])]);sizes=np.diff(np.r_[starts,len(e)]);assert sizes.max()<=8
  rows=np.repeat(np.arange(len(starts)),sizes);cols=np.arange(len(e))-np.repeat(starts,sizes);ids=np.full((len(starts),8),-1,np.int64);ids[rows,cols]=e
  return t[starts],ids,p[starts]
 send_idx,send_eids,send_dest=packets(src==rank,dst)
 recv_idx,recv_eids,recv_origins=packets(dst==rank,src)
 rr,cc=np.nonzero(recv_eids>=0);ee=recv_eids[rr,cc];order=np.argsort(ee,kind='stable');rr=rr[order];cc=cc[order];ee=ee[order]
 groups=[]
 if len(ee):
  values,starts,sizes=np.unique(ee,return_index=True,return_counts=True)
  groups=[(int(e),rr[start:start+n].tolist(),cc[start:start+n].tolist()) for e,start,n in zip(values,starts,sizes)]
 partial_origins=recv_origins[rr];return_order=np.argsort(partial_origins,kind='stable')
 mine=src==rank;e=experts[mine];t=local[mine];remote=dst[mine];order=np.lexsort((t,e,remote));e=e[order];t=t[order];remote=remote[order]
 combine=[(t[e==expert].tolist(),np.flatnonzero(e==expert).tolist()) for expert in np.unique(e)]
 return dict(send_counts=np.bincount(send_dest,minlength=world).tolist(),recv_counts=np.bincount(recv_origins,minlength=world).tolist(),send_idx=send_idx.tolist(),send_eids=send_eids.tolist(),groups=groups,return_order=return_order.tolist(),return_counts=np.bincount(partial_origins,minlength=world).tolist(),return_recv_counts=np.bincount(remote,minlength=world).tolist(),combine=combine)
