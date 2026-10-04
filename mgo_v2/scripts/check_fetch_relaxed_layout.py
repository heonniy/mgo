"""Independent CPU exchange simulation for A2 return-weight alignment."""
from fetch_relaxed_common import *
from env_offload_layout import plan_layout

def check(world):
 rng=np.random.default_rng(130+world);counts=[r+2 for r in range(world)];offset=np.r_[0,np.cumsum(counts)];n=sum(counts);orig=np.repeat(np.arange(world),counts)
 selected=np.array([rng.choice(128,8,replace=False) for _ in range(n)],np.int64);weights=rng.random((n,8),dtype=np.float32);weights/=weights.sum(1,keepdims=True);dst=selected%world
 layouts=[plan_layout(selected,np.full(n,8),dst,orig,counts,r) for r in range(world)];sent_weights=[];sent_tokens=[];return_weights=[]
 for r,e in enumerate(layouts):
  local=selected[offset[r]:offset[r+1]];w=weights[offset[r]:offset[r+1]];dense=np.zeros((counts[r],128),np.float32);np.put_along_axis(dense,local,w,axis=1)
  ids=np.array(e['send_eids']);idx=np.array(e['send_idx']);sent_weights.append(dense[idx[:,None],ids.clip(0)]*(ids>=0));sent_tokens.append(offset[r]+idx)
  rw=np.zeros(sum(e['return_recv_counts']),np.float32)
  for expert,(tokens,positions) in zip(np.unique(local),e['combine']):
   tokens=np.array(tokens);match=local[tokens]==expert;assert np.all(match.sum(1)==1);rw[np.array(positions)]=w[tokens,match.argmax(1)]
  return_weights.append(rw)
 raw=[];weighted=[]
 for r,e in enumerate(layouts):
  received=[];rw=[]
  for src,send in enumerate(layouts):
   lo=sum(send['send_counts'][:r]);hi=lo+send['send_counts'][r];received.append(sent_tokens[src][lo:hi]);rw.append(sent_weights[src][lo:hi])
  received=np.concatenate(received);rw=np.concatenate(rw);a=[];b=[]
  for expert,rows,cols in e['groups']:
   rows=np.array(rows);cols=np.array(cols);value=received[rows].astype(np.float32)+np.float32(expert/129);a.extend(value.tolist());b.extend((value*rw[rows,cols]).tolist())
  order=np.array(e['return_order']);raw.append(np.array(a,np.float32)[order]);weighted.append(np.array(b,np.float32)[order])
 for r,e in enumerate(layouts):
  a=[];b=[]
  for src,send in enumerate(layouts):
   lo=sum(send['return_counts'][:r]);hi=lo+send['return_counts'][r];a.append(raw[src][lo:hi]);b.append(weighted[src][lo:hi])
  a=np.concatenate(a)*return_weights[r];b=np.concatenate(b);assert np.array_equal(a,b)
 return dict(world=world,unequal_prefill_rank_counts=counts,returned_partials=n*8,exact_weighted_partial_equality=True)
assert eligible(dict(total_fetches=100000,H2D_bytes=100000*9437184),dict(total_fetches=100100,H2D_bytes=100100*9437184))
assert not eligible(dict(total_fetches=100000,H2D_bytes=100000*9437184),dict(total_fetches=100101,H2D_bytes=100101*9437184))
write(PACKET/'layout_validation.json',dict(status='PASS',CPU_only=True,threshold_boundary_inclusive=True,checks=[check(4),check(8)]));print('A2 routing/weight alignment CPU fixture PASS')
