"""Freeze source-derived matrices and expert row bundles without GPU capture."""
import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
PACKET=P/'experiments/critical_path_admission_followup_20261006'
ROOT=Path('/home/hwlee/mgo-results/critical_path_admission_followup_20261006')
OLD=Path('/home/hwlee/mgo-results/policy_regime_20261005')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def features(m):
 a=np.array(m);s=a.sum(1);r=a.sum(0)
 return dict(total_packets=int(a.sum()),send_by_rank=s.tolist(),recv_by_rank=r.tolist(),incident_by_rank=(s+r).tolist(),max_send=int(max(s)),max_recv=int(max(r)),max_incident=int(max(s+r)),max_peer_edge=int(a.max()),active_directed_edges=int(np.count_nonzero(a)),active_peers_by_rank=np.count_nonzero(a,axis=1).tolist(),send_cv=float(s.std()/s.mean()),recv_cv=float(r.std()/r.mean()))
def scaled(m,total):
 a=np.array(m,dtype=float);z=a.ravel()/a.sum()*total;x=np.floor(z).astype(int)
 for i in np.argsort(-(z-x),kind='stable')[:total-int(x.sum())]:x[i]+=1
 return x.reshape(4,4).tolist()
def main():
 ROOT.mkdir(exist_ok=True);sources={};records=[]
 for c in ('C30','C60'):
  for pol in ('BR','FCA'):
   folder=OLD/c/f'POLICY_REGIME_PROFILE_{c}_NSYS2025_V3_OPT_PF_OVERLAP_{pol}_B128_H64'
   ranks=[]
   for r in range(4):
    f=folder/f'communication_rank{r}.json';sources[str(f)]=sha(f);ranks.append(json.loads(f.read_text())['events'])
   for i in range(3072):
    m=np.array([ranks[r][i]['send_token_rows'] for r in range(4)])
    assert all(m[s,d]==ranks[d][i]['recv_token_rows'][s] for s in range(4) for d in range(4))
    np.fill_diagonal(m,0)
    records.append(dict(cache=c,policy=pol,event=i+48,matrix=m.tolist(),**features(m)))
 br=[x for x in records if x['policy']=='BR'];volumes=[int(np.quantile([x['total_packets'] for x in br],q,method='nearest')) for q in (.5,.9,.99)]
 # Quantiles may coincide at the all-peers ceiling: keep three distinct observed
 # levels nearest median/p90/p99, choosing the next smaller distinct count.
 observed=sorted({x['total_packets'] for x in br});used=set()
 for i,v in enumerate(volumes):
  while v in used:v=max(x for x in observed if x<v)
  volumes[i]=v;used.add(v)
 shapes=[]
 balanced=np.ones((4,4),int)-np.eye(4,dtype=int)
 pair=np.zeros((4,4),int)
 for s,d in ((0,1),(1,0),(2,3),(3,2)):pair[s,d]=1
 dest=np.zeros((4,4),int);dest[1:,0]=1
 src=dest.T
 for v in volumes:
  for name,m in [('BALANCED',balanced),('PAIR_HOT',pair),('DEST_HOT',dest),('SRC_HOT',src)]:
   a=scaled(m,v);shapes.append(dict(id=f'V{v}_{name}',shape=name,matrix=a,**features(a)))
  for pol in ('BR','FCA'):
   item=min((x for x in records if x['policy']==pol),key=lambda x:(abs(x['total_packets']-v),x['cache'],x['event']))
   a=scaled(item['matrix'],v)
   shapes.append(dict(id=f'V{v}_TRACE_{pol}',shape='TRACE_'+pol,matrix=a,source=item,scaled_to_matched_volume=True,**features(a)))
   shapes.append(dict(id=f'V{v}_RAW_TRACE_{pol}',shape='RAW_TRACE_'+pol,matrix=item['matrix'],source=item,**features(item['matrix'])))
 inp=OLD/'C30/inputs_B128_H64';sel=np.load(inp/'selected.npy',mmap_mode='r');off=np.load(inp/'offsets.npy',mmap_mode='r')
 sources[str(inp/'selected.npy')]=sha(inp/'selected.npy');sources[str(inp/'offsets.npy')]=sha(inp/'offsets.npy')
 counts=[];bundles=[]
 for event in range(48,3120):
  lo,hi=off[event:event+2];ns=np.bincount(sel[lo:hi].ravel(),minlength=128);counts.extend(ns[ns>0].tolist())
  if event in (48,49,64,128,1024,2048,3071):
   active=ns[ns>0].tolist()
   for k in (2,4,8):bundles.append(dict(event=event,rows=active[:k]))
 maxrows=max(counts);rows=[2**i for i in range(max(9,int(np.ceil(np.log2(maxrows))))+1)]
 data=dict(status='FROZEN',plan_commit='894ed896292d3ff628f3db2dbc34a04d45461393',gpus=[0,1,4,5],volumes=volumes,volume_quantiles=[.5,.9,.99],volume_note='Distinct observed levels nearest median/p90/p99; colliding quantiles use next lower observed count.',shapes=shapes,arrival_matrices=[shapes[0],next(x for x in shapes if x['shape']=='RAW_TRACE_FCA')],tau_rows=rows,bundles=bundles,trace_expert_rows=dict(max=maxrows,p99=float(np.quantile(counts,.99))),source_sha256=sources,scope='Remote edges only: self copies excluded in every causal comparison. Raw trace shapes also retained unscaled. Bundles use real global expert demand counts, not claimed to be one rank scheduling order.')
 write(ROOT/'inputs.json',data);write(PACKET/'MICROBENCH_MANIFEST.json',data)
 print(json.dumps(dict(volumes=volumes,shapes=len(shapes),tau_rows=rows,bundles=len(bundles))))
if __name__=='__main__':main()
