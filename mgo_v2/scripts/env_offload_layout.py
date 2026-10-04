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

def add_coslot_layout(event,world):
 """Expand token->rank packets into CoSLoT-style (token, expert) routes.

 The placement/substitution schedule is unchanged.  Only the transport view is
 changed: each valid expert in a packet becomes one route row, matching
 MoE-Infinity-EP-coslot's fast-A2A semantics.
 """
 e=dict(event)
 if 'coslot_send_idx' in e:return e
 send_idx=np.asarray(e['send_idx'],dtype=np.int64)
 send_eids=np.asarray(e['send_eids'],dtype=np.int64).reshape(-1,8)
 send_counts=np.asarray(e['send_counts'],dtype=np.int64)
 assert len(send_idx)==len(send_eids)==int(send_counts.sum())
 packet_dest=np.repeat(np.arange(world,dtype=np.int64),send_counts)
 sr,sc=np.nonzero(send_eids>=0)
 route_send_idx=send_idx[sr]
 route_send_eids=send_eids[sr,sc]
 route_dest=packet_dest[sr]
 coslot_send_counts=np.bincount(route_dest,minlength=world).tolist()

 recv_counts=np.asarray(e['recv_counts'],dtype=np.int64)
 assert int(recv_counts.sum())>=0
 packet_src=np.repeat(np.arange(world,dtype=np.int64),recv_counts)
 records=[]
 for expert,rows,cols,slot in e['groups']:
  for row,col in zip(np.asarray(rows,dtype=np.int64),np.asarray(cols,dtype=np.int64)):
   records.append((int(row),int(col),int(expert),int(slot)))
 records.sort(key=lambda x:(x[0],x[1]))
 assert len({(r,c) for r,c,_,_ in records})==len(records)
 if records:
  packet_rows=np.asarray([r for r,_,_,_ in records],dtype=np.int64)
  assert packet_rows.min()>=0 and packet_rows.max()<len(packet_src)
  route_origins=packet_src[packet_rows]
 else:
  route_origins=np.empty(0,dtype=np.int64)
 route_of={(r,c):i for i,(r,c,_,_) in enumerate(records)}
 coslot_groups=[];part_routes=[]
 for expert,rows,cols,slot in e['groups']:
  routes=[route_of[(int(r),int(c))] for r,c in zip(rows,cols)]
  coslot_groups.append((int(expert),routes,int(slot)));part_routes.extend(routes)
 part_routes=np.asarray(part_routes,dtype=np.int64)
 if len(part_routes):
  assert len(np.unique(part_routes))==len(part_routes)==len(records)
  part_origins=route_origins[part_routes]
  coslot_return_order=np.lexsort((part_routes,part_origins)).astype(np.int64)
 else:
  part_origins=np.empty(0,dtype=np.int64);coslot_return_order=np.empty(0,dtype=np.int64)
 coslot_recv_counts=np.bincount(route_origins,minlength=world).tolist()
 coslot_return_counts=np.bincount(part_origins,minlength=world).tolist()
 assert coslot_recv_counts==coslot_return_counts
 e.update(
  coslot_combine=[(route_send_idx[route_send_eids==expert].tolist(),np.flatnonzero(route_send_eids==expert).tolist()) for expert in np.unique(route_send_eids)],
  coslot_send_idx=route_send_idx.tolist(),
  coslot_send_eids=route_send_eids.tolist(),
  coslot_send_counts=coslot_send_counts,
  coslot_recv_counts=coslot_recv_counts,
  coslot_groups=coslot_groups,
  coslot_return_order=coslot_return_order.tolist(),
  coslot_return_counts=coslot_return_counts,
  coslot_return_recv_counts=coslot_send_counts,
 )
 return e
