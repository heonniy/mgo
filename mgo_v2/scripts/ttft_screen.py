"""Exact cold-prefill placement feature screen, with generic-controller audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
from ttft_common import *
import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
from br_carep_cpu import balanced_assignment
import mgo_v2  # Initialize controller package before its legacy Policy dependency.
from env_offload_policy import seed_rng,Policy
from la_placement import load_assignment
import concurrent.futures,multiprocessing,time

def order_for(batch,seed):return np.random.default_rng(seed).permutation(2048)[:4*batch]
@njit(cache=True)
def histogram(selected,batch):
 d=np.zeros((128,4),np.int64)
 for t in range(len(selected)):
  for e in selected[t]:d[e,t//(batch*512)]+=1
 return d
@njit(cache=True)
def features(selected,assignment,batch):
 rows=np.zeros(4,np.int64);packets=np.zeros((4,4),np.int64)
 for t in range(len(selected)):
  origin=t//(batch*512);mask=0
  for e in selected[t]:
   dst=assignment[e];rows[dst]+=1;mask|=1<<dst
  for r in range(4):packets[origin,r]+=(mask>>r)&1
 return rows,packets

def pack_pool():
 target=ROOT/'pool';receipt=target/'receipt.json'
 if receipt.exists():assert json.loads(receipt.read_text())['status']=='PASS';return
 assert json.loads((ROOT/'capture_status.json').read_text())['status']=='PASS'
 target.mkdir(exist_ok=True)
 shapes={'selected':((2048,48,512,8),'uint8'),'weights':((2048,48,512,8),'float32'),'gates':((2048,48,128),'float32'),'first_tokens':((2048,),'int64')}
 dst={k:np.lib.format.open_memmap(target/(k+'.npy'),mode='w+',shape=s,dtype=t) for k,(s,t) in shapes.items()}
 for i,g in enumerate(GPUS):
  src=ROOT/'capture'/f'gpu{g}';r=json.loads((src/'receipt.json').read_text());assert r['status']=='PASS' and r['request_ids']==list(range(i*512,(i+1)*512))
  assert r['request_manifest_sha256']==sha(ROOT/'requests.json')
  for k in dst:
   f=src/(k+'.npy');assert sha(f)==r['files'][f.name];dst[k][i*512:(i+1)*512]=np.load(f,mmap_mode='r')
 for x in dst.values():x.flush()
 write(receipt,dict(status='PASS',request_manifest_sha256=sha(ROOT/'requests.json'),files={k:sha(target/(k+'.npy')) for k in dst}))

def summarize(rs,ps,assignments):
 return dict(sum_max_rank_rows=int(sum(r.max() for r in rs)),sum_min_rank_rows=int(sum(r.min() for r in rs)),expert_rows_by_rank=np.sum(rs,axis=0).tolist(),remote_packets=int(sum(p.sum()-np.trace(p) for p in ps)),remote_expert_routes=None,assignment_sha256=__import__('hashlib').sha256(np.array(assignments,np.int8).tobytes()).hexdigest(),mandatory_H2D_copies_by_rank=np.bincount(np.array(assignments).ravel()[np.array(assignments).ravel()>=0],minlength=4).tolist())

def job(spec):
 batch,wseed=spec;path=ROOT/'screen'/f'B{batch}_w{wseed}.json'
 if path.exists():assert json.loads(path.read_text())['status']=='PASS';return str(path)
 import psutil
 assert psutil.virtual_memory().available>256*2**30
 start=time.monotonic();order=order_for(batch,wseed);source=np.load(ROOT/'pool/selected.npy',mmap_mode='r');routes=np.ascontiguousarray(source[order].transpose(1,0,2,3).reshape(48,4*batch*512,8))
 demands=[histogram(r,batch) for r in routes];active=[np.flatnonzero(d.sum(1)) for d in demands]
 assignments={};zeros=np.zeros(48*128,np.int16)
 for policy in ('CA','OLD_CA','LA'):
  a=[]
  for layer,(d,act) in enumerate(zip(demands,active)):
   slots=np.repeat(np.arange(4),[len(act)//4+(r<len(act)%4) for r in range(4)])
   if policy=='CA':dst=balanced_assignment(d,act,4,False)
   elif policy=='OLD_CA':
    costs=d[act].sum(1)[:,None]-d[act];rr,cc=linear_sum_assignment(costs[:,slots]);dst=np.empty(len(act),np.int64);dst[rr]=slots[cc]
   else:dst=load_assignment(d,act,zeros,layer)
   full=np.full(128,-1,np.int8);full[act]=dst;a.append(full)
  assignments[policy]=a
 candidates={}
 for policy,a in assignments.items():
  measurements=[features(r,x,batch) for r,x in zip(routes,a)];candidates[policy]=summarize([x[0] for x in measurements],[x[1] for x in measurements],a)
 br=[];br_assignments=[]
 for seed in range(32):
  seed_rng(seed);a=[]
  for d,act in zip(demands,active):
   x=np.full(128,-1,np.int8);x[act]=balanced_assignment(d,act,4,True);a.append(x)
  measurements=[features(r,x,batch) for r,x in zip(routes,a)];br.append(dict(placement_seed=seed,**summarize([x[0] for x in measurements],[x[1] for x in measurements],a)));br_assignments.append(a)
 # A full 48-layer generic Policy replay independently validates the shortcut.
 audit=[]
 if wseed==0:
  weights=np.load(ROOT/'pool/weights.npy',mmap_mode='r');gates=np.load(ROOT/'pool/gates.npy',mmap_mode='r');origins=np.repeat(np.arange(4,dtype=np.int8),batch*512)
  for cache in (30,60):
   slots=48*128*cache//100;caps=[slots//4+(r<slots%4) for r in range(4)]
   for policy,kind in [('BR',0),('CA',1),('OLD_CA',3),('LA',4)]:
    c=Policy(caps,np.zeros((48,128,128),np.float32),False,kind,0);expected=br_assignments[0] if policy=='BR' else assignments[policy]
    for layer in range(48):
     w=np.ascontiguousarray(weights[order,layer].reshape(-1,8));out=c.apply(layer,routes[layer],w,origins,gates[order[-1],layer],np.zeros((128,4),np.int32))
     act=active[layer];assert np.array_equal(c.primary[layer*128+act],expected[layer][act]),(cache,policy,layer)
     rs,pack=features(routes[layer],expected[layer],batch);assert int(out[6][30])==int(pack.sum()-np.trace(pack))
     assert len(out[5])==len(act) and all(f[-1]==0 for f in out[5])
    audit.append(dict(cache=cache,policy=policy,layers=48,status='PASS'))
 write(path,dict(status='PASS',batch=batch,workload_seed=wseed,requests=order.tolist(),candidates=candidates,BR=br,seed_activity=dict(initial_cache='empty for all seeds',candidate_placement_seed_inactive=True,BR_assignment_hashes=len({x['assignment_sha256'] for x in br})),generic_controller_audit=audit,elapsed_seconds=time.monotonic()-start));return str(path)

def main():
 pack_pool();(ROOT/'screen').mkdir(exist_ok=True)
 started=time.time();workers=min(16,max(1,len(os.sched_getaffinity(0))//8))
 # Compile once before spawning workers; each worker remains single-threaded.
 histogram(np.tile(np.arange(8,dtype=np.uint8),(2048,1)),1);features(np.tile(np.arange(8,dtype=np.uint8),(2048,1)),np.arange(128,dtype=np.int8)%4,1)
 state=dict(status='RUNNING',workers=workers,completed=[],started_unix=started)
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  for path in pool.map(job,[(b,s) for b in (16,128) for s in range(32)]):
   state['completed'].append(path);write(ROOT/'screen_status.json',state);print('completed',len(state['completed']),'/64',flush=True)
 shortlist=[]
 for cache in (30,60):
  for batch in (16,128):
   for policy in ('CA','OLD_CA','LA'):
    rows=[]
    for seed in range(32):
     x=json.loads((ROOT/'screen'/f'B{batch}_w{seed}.json').read_text());c=x['candidates'][policy]
     for b in x['BR']:
      load=1-c['sum_max_rank_rows']/b['sum_max_rank_rows'];comm=1-c['remote_packets']/b['remote_packets']
      rows.append(dict(cache=cache,batch=batch,policy=policy,workload_seed=seed,placement_seed=b['placement_seed'],load_proxy_gain=load,remote_packet_proxy_gain=comm,BR=b,candidate=c))
    rows.sort(key=lambda x:(-(x['load_proxy_gain'] if policy=='LA' else x['remote_packet_proxy_gain']),-(x['remote_packet_proxy_gain'] if policy=='LA' else x['load_proxy_gain']),x['workload_seed'],x['placement_seed']))
    shortlist.extend(rows[:8])
 state.update(status='PASS',finished_unix=time.time());write(ROOT/'screen_status.json',state)
 write(PACKET/'TTFT_S0_SHORTLIST.json',dict(status='PASS',rows=shortlist,ranking='CA/OLD_CA: remote token-rank packet reduction, then max-rank row reduction. LA: max-rank row reduction, then packets. Exact features, no latency prediction or speed claim.',cold_prefill_cache_equivalence='Each layer is encountered once from empty cache; all current-layer experts are mandatory misses. Capacity changes victim slots, not current-layer demand/placement. Both cache sizes validated by generic48-layer replay on workload0 for both batches and all4 policies.',generic_audit_paths=[str(ROOT/'screen'/f'B{b}_w0.json') for b in (16,128)],seed_activity='Candidates deterministic; their placement seed inactive. BR admission assignment is seeded, so the paired BR-relative seed axis remains active.',screen=state))
if __name__=='__main__':main()
