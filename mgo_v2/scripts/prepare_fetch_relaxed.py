"""Offline common-route schedules; never ask a live model to select experts."""
from fetch_relaxed_common import *
from env_offload_policy import Policy
from env_offload_layout import plan_layout
import gzip,pickle,time,resource

def build(win,root=ROOT,packet=PACKET,policies=(('BR',0),('CA',1)),compresslevel=9):
 world=win['world'];key=f"R{world}_s{win['sample_seed']}_d{win['dp_seed']}_b{win['BR_seed']}";dest=root/'plans'/key
 if (dest/'receipt.json').exists():return json.loads((dest/'receipt.json').read_text())
 dest.mkdir(parents=True,exist_ok=False);pool=PrefixPool('ShareGPT');order=np.array(win['request_ids']);a=pool.pack(order,world,32)
 assert hashlib.sha256(a['selected'].tobytes()).hexdigest()==win['route_sha256'] and hashlib.sha256(a['gates'].tobytes()).hexdigest()==win['gate_sha256']
 cap=np.array([1843//world+(r<1843%world) for r in range(world)],np.int32)
 source=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/ShareGPT_requests.json');requests=json.loads(source.read_text())['requests'];teacher=np.load(pool.path/'generated_tokens.npy',mmap_mode='r')[order,:64]
 write(dest/'requests.json',dict(ranks=[[requests[i] for i in order[r*32:(r+1)*32]] for r in range(world)],request_ids=order.tolist(),source_sha256=sha(source)));np.save(dest/'teacher_tokens.npy',teacher);proofs={}
 for name,index in policies:
  pol=Policy(cap,np.zeros((48,128,128),np.float32),False,index,win['BR_seed'] if index==0 else 42);events=[[] for _ in range(world)];metrics=[];traffic=np.zeros((3120,2,world,2),np.int64);fanout=np.zeros((3120,world),np.int64)
  for i in range(3120):
   lo,hi=a['offsets'][i:i+2];orig=a['prefill_origins'] if i<48 else a['decode_origins'];selected=a['selected'][lo:hi];weights=a['weights'][lo:hi]
   targets,effective,masses,lengths,destinations,fetches,row=pol.apply(i,selected,weights,orig,a['gates'][i],np.zeros((128,world),np.int32));metrics.append(row)
   counts=np.bincount(orig,minlength=world);offsets=np.r_[0,np.cumsum(counts)]
   for rank in range(world):
    e=plan_layout(effective,lengths,destinations,orig,counts,rank);s,t=offsets[rank:rank+2];local=selected[s:t];weight=weights[s:t]
    e.update(layer=i%48,targets=targets,selected=local.copy(),weights=weight.copy(),fetches=[(k,slot,victim,rep) for r,k,slot,victim,rep in fetches if r==rank]);e['groups']=[(expert,rows,cols,int(np.flatnonzero(pol.slots[rank]==i%48*128+expert)[0])) for expert,rows,cols in e['groups']]
    return_weights=np.zeros(sum(e['return_recv_counts']),np.float32)
    for expert,(tokens,positions) in zip(np.unique(local),e['combine']):
     tokens=np.array(tokens,np.int64);positions=np.array(positions,np.int64);matches=local[tokens]==expert;assert np.all(matches.sum(1)==1)
     return_weights[positions]=weight[tokens,np.argmax(matches,axis=1)]
    e['return_weights']=return_weights;events[rank].append(e)
    for j,(send,recv) in enumerate([('send_counts','recv_counts'),('return_counts','return_recv_counts')]):traffic[i,j,rank]=[sum(e[send])-e[send][rank],sum(e[recv])-e[recv][rank]]
    fanout[i,rank]=sum(n>0 for r,n in enumerate(e['send_counts']) if r!=rank)
  expected=win[name];rows=np.array(metrics)
  assert int(traffic[:,:,:,0].sum())*4096==expected['full']['peer_bytes'] and int(traffic.sum(3).max(2).sum())*4096==expected['Critical_total']
  assert int((rows[:,20]+rows[:,21]).sum())*9437184==expected['full']['H2D_bytes']
  state=hashlib.sha256()
  for v in [pol.slots,pol.owner,pol.primary,pol.last,pol.seen,pol.lost,pol.birth,pol.reuses,np.array([0,0],np.int64)]:state.update(v.tobytes())
  assert state.hexdigest()==expected['final_state_sha256']
  proofs[name]=[]
  for rank in range(world):
   raw=events[rank];path=dest/f'{name}_rank{rank}.pkl.gz'
   with gzip.open(path,'wb',compresslevel=compresslevel) as f:pickle.dump(raw,f,protocol=4)
   h=hashlib.sha256()
   for e in raw:h.update(e['selected'].tobytes());h.update(e['weights'].tobytes())
   proof=dict(status='PASS',rank=rank,policy=name,schedule_sha256=sha(path),route_weight_sha256=h.hexdigest(),state_hash=hashlib.sha256(pol.slots[rank,:cap[rank]].tobytes()).hexdigest(),H2D_bytes=sum(len(e['fetches']) for e in raw)*9437184,dispatch_bytes=int(traffic[:,0,rank,0].sum())*4096,return_bytes=int(traffic[:,1,rank,0].sum())*4096,weight_bytes=int(traffic[:,0,rank,0].sum())*16,capacity=int(cap[rank]))
   proofs[name].append(proof)
  np.save(dest/(name+'_traffic.npy'),traffic);np.save(dest/(name+'_metrics.npy'),rows);np.save(dest/(name+'_fanout.npy'),fanout)
  del events
 assert all(proofs['BR'][r]['route_weight_sha256']==proofs[name][r]['route_weight_sha256'] for r in range(world) for name in proofs)
 receipt=dict(status='PASS',id=key,world=world,path=str(dest),winner=win,proofs=proofs,teacher_sha256=sha(dest/'teacher_tokens.npy'),request_manifest_sha256=sha(dest/'requests.json'),CPU_peer_and_critical_match=True,horizon=64)
 write(dest/'receipt.json',receipt);write(packet/(key+'_plan_validation.json'),receipt);print('FROZEN_PLAN',key,flush=True);return receipt

def main():
 unique={}
 for win in json.loads((PACKET/'winners.json').read_text()):unique.setdefault((win['world'],win['sample_seed'],win['dp_seed'],win['BR_seed']),win)
 plans=[build(win) for _,win in sorted(unique.items())];write(ROOT/'physical_plans.json',plans)
if __name__=='__main__':main()
