"""Two frozen routing prescreens, then exact fetch-matched decode64 replay."""
from fetch_matched_common import *
import time,resource,json
from numba import njit
from fetch_matched_replay import replay_critical
from br_carep_cpu import replay
from run_br_cpu import summarize
import run_ca_stress_cpu as manager

@njit(cache=True)
def traffic_for(selected,origins,dest,world):
 t=np.zeros((2,world,2),np.int64)
 for i in range(len(selected)):
  src=origins[i];bits=0
  for k in range(8):
   dst=dest[selected[i,k]]
   if dst!=src:t[1,dst,0]+=1;t[1,src,1]+=1;bits|=1<<dst
  for dst in range(world):
   if (bits>>dst)&1:t[0,src,0]+=1;t[0,dst,1]+=1
 return t

@njit(cache=True)
def proxy(routes,sample,orders,world):
 out=np.zeros((len(orders),2),np.float64);origins=np.arange(len(sample))//32
 for dp in range(len(orders)):
  np.random.seed(42)
  for event in range(len(routes)):
   selected=routes[event,sample[orders[dp]]];d=np.zeros((128,world),np.int64)
   for i in range(len(sample)):
    for k in range(8):d[selected[i,k],origins[i]]+=1
   active=np.flatnonzero(d.sum(1)>0);n=len(active);q=np.array([n//world+(r<n%world) for r in range(world)])
   gaps=np.zeros(n,np.int64)
   for j in range(n):v=np.sort(d[active[j]]);gaps[j]=v[-1]-v[-2]
   ca=np.full(128,-1,np.int64);remaining=q.copy()
   for j in np.argsort(-gaps,kind='mergesort'):
    e=active[j];best=-1;value=-1
    for r in range(world):
     if remaining[r]>0 and d[e,r]>value:best=r;value=d[e,r]
    ca[e]=best;remaining[best]-=1
   shuffled=active.copy();np.random.shuffle(shuffled);br=np.full(128,-1,np.int64);at=0
   for r in range(world):
    for _ in range(q[r]):br[shuffled[at]]=r;at+=1
   a=traffic_for(selected,origins,br,world);b=traffic_for(selected,origins,ca,world)
   out[dp,0]+=(a[:,:,0].sum()-b[:,:,0].sum())*4096
   for collective in range(2):out[dp,1]+=((a[collective,:,0]+a[collective,:,1]).max()-(b[collective,:,0]+b[collective,:,1]).max())*4096
 return out/len(routes)

def execute(a,world,policy,seed):
 cap=np.array([1843//world+(r<1843%world) for r in range(world)],np.int64);sim=np.zeros((48,128,128),np.float32)
 result=replay_critical(a['selected'],a['weights'],a['offsets'],a['prefill_origins'],a['decode_origins'],a['gates'],sim,cap,64,True,False,policy,seed)
 rows,fetch=result[:2];traffic=result[-1];full=summarize(rows,fetch,1843,64);h=hashlib.sha256()
 for arr in result[2:-1]:h.update(arr.tobytes())
 assert int(traffic[:,:,:,0].sum())*4096==full['peer_bytes']
 assert np.array_equal(traffic[:,:,:,0].sum(2),traffic[:,:,:,1].sum(2))
 assert full['max_event_rank_fetch_imbalance']<=1 and full['peak_resident_copies']<=1843
 return dict(policy=['BR','CA'][policy],seed=seed,full=full,decode=summarize(rows[48:],fetch[48:],1843,64),Critical_total=int(traffic.sum(3).max(2).sum())*4096,max_rank_whole_run_send_recv=int(traffic.sum((0,1,3)).max())*4096,rank_send_bytes=(traffic[:,:,:,0].sum((0,1))*4096).tolist(),rank_recv_bytes=(traffic[:,:,:,1].sum((0,1))*4096).tolist(),dispatch_bytes=int(traffic[:,0,:,0].sum())*4096,return_bytes=int(traffic[:,1,:,0].sum())*4096,weight_bytes_A3=int(traffic[:,0,:,0].sum())*16,final_state_sha256=h.hexdigest()),result

def worker(tasks,results,cpu):
 os.sched_setaffinity(0,{cpu});resource.setrlimit(resource.RLIMIT_AS,(96*2**30,96*2**30));resource.setrlimit(resource.RLIMIT_DATA,(8*2**30,8*2**30))
 pool=PrefixPool('ShareGPT');routes=np.stack([pool.a['decode_selected'][(l%4)*21,l] for l in range(48)]);orders={w:np.array([np.random.default_rng(d).permutation(w*32) for d in range(64)],np.int64) for w in [4,8]}
 while True:
  job=tasks.get()
  if job is None:return
  begin=time.monotonic()
  try:
   w=job['world'];sample=job['sample_seed'];path=ROOT/job['kind']/f"R{w}_s{sample}";path=path.with_name(path.name+(f"_d{job['dp_seed']}" if job['kind']=='B' else '')+'.json')
   if not path.exists():
    if job['kind']=='A':
     ids=np.random.default_rng(sample).permutation(pool.n)[:w*32];scores=proxy(routes,ids,orders[w],w);out=dict(status='PASS',job=job,scores=scores.tolist())
    else:
     order=pool.candidate(w,32,sample,job['dp_seed']);a=pool.pack(order,w,32);ca,_=execute(a,w,1,42);br=[execute(a,w,0,seed)[0] for seed in SEEDS]
     out=dict(status='PASS',job=job,request_ids=order.tolist(),manifest_sha256=digest(order.tolist()),route_sha256=hashlib.sha256(a['selected'].tobytes()).hexdigest(),gate_sha256=hashlib.sha256(a['gates'].tobytes()).hexdigest(),CA=ca,BR=br)
    out.update(seconds=time.monotonic()-begin,source_sha256=sha(Path(__file__)),pool_receipt_sha256=sha(pool.path/'receipt.json'));write(path,out)
   old=json.loads(path.read_text());assert old['status']=='PASS' and old['source_sha256']==sha(Path(__file__))
   results.put(dict(status='PASS',job=job,path=str(path),seconds=time.monotonic()-begin))
  except BaseException as exc:results.put(dict(status='FAIL',job=job,error=repr(exc)));return

def validation(pool):
 w=4;a=pool.pack(pool.candidate(w,32,0,0),w,32);checks=[]
 cap=np.array([1843//w+(r<1843%w) for r in range(w)],np.int64);sim=np.zeros((48,128,128),np.float32)
 for policy in [0,1]:
  _,actual=execute(a,w,policy,42);expected=replay(a['selected'],a['weights'],a['offsets'],a['prefill_origins'],a['decode_origins'],a['gates'],sim,cap,64,True,False,policy,42)
  assert all(np.array_equal(x,y) for x,y in zip(actual[:-1],expected));checks.append(dict(policy=policy,all_original_arrays_equal=True))
 write(PACKET/'CPU_validation.json',dict(status='PASS',horizon=64,reference_policy_unchanged=True,checks=checks))

def select(retained):
 eligible=[];checked=0
 for job in retained:
  path=ROOT/'B'/f"R{job['world']}_s{job['sample_seed']}_d{job['dp_seed']}.json";d=json.loads(path.read_text());ca=d['CA']
  for br in d['BR']:
   checked+=1;a,b=br['full'],ca['full']
   if a['total_fetches']!=b['total_fetches'] or a['H2D_bytes']!=b['H2D_bytes']:continue
   peer=a['peer_bytes']-b['peer_bytes'];crit=br['Critical_total']-ca['Critical_total']
   eligible.append(dict(world=job['world'],sample_seed=job['sample_seed'],dp_seed=job['dp_seed'],BR_seed=br['seed'],peer_gain_bytes=peer,peer_gain_relative=peer/a['peer_bytes'],critical_gain_bytes=crit,critical_gain_relative=crit/br['Critical_total'],reload_equal=a['reload_fetches']==b['reload_fetches'],fetches=a['total_fetches'],H2D_bytes=a['H2D_bytes'],candidate_path=str(path),candidate_sha256=sha(path),request_ids=d['request_ids'],CA=ca,BR=br))
 winners=[];alternatives=[]
 for world in [4,8]:
  group=[r for r in eligible if r['world']==world]
  for objective in ['Peer-best','Critical-best']:
   def key(r):
    values=(r['peer_gain_bytes'],r['peer_gain_relative'],r['critical_gain_bytes'],r['reload_equal']) if objective=='Peer-best' else (r['critical_gain_bytes'],r['critical_gain_relative'],r['peer_gain_bytes'],r['reload_equal'])
    return (*values,-r['sample_seed'],-r['dp_seed'],-r['BR_seed'])
   ranked=sorted(group,key=key,reverse=True)
   if ranked:winners.append(dict(objective=objective,**ranked[0]))
   alternatives.append(dict(world=world,objective=objective,top10=ranked[:10]))
 write(PACKET/'top10.json',alternatives);write(PACKET/'winners.json',winners)
 write(PACKET/'fetch_match_validation.json',dict(status='PASS' if len(winners)==4 else 'NO_ELIGIBLE_PAIR_FOR_SOME_R',prescreen_pairs=65536,retained=len(retained),exact_policy_replays=len(retained)*9,BR_CA_pairs_checked=checked,eligible_pairs=len(eligible),eligible_per_R={w:sum(r['world']==w for r in eligible) for w in [4,8]},hard_match='exact total fetches AND H2D bytes, never rounded',horizon=64))
 return winners

def main():
 ROOT.mkdir(exist_ok=True);pool=PrefixPool('ShareGPT');assert pool.n==2048
 validation(pool);manager.worker=worker;manager.ROOT=ROOT;manager.PACKET=PACKET
 jobs=[dict(kind='A',dataset='ShareGPT',world=w,batch=32,sample_seed=s) for w in [4,8] for s in range(512)];manager.drive(jobs,'A')
 retained=[]
 for w in [4,8]:
  scores=np.array([json.loads((ROOT/'A'/f'R{w}_s{s}.json').read_text())['scores'] for s in range(512)]);assert scores.shape==(512,64,2)
  indices=set()
  for objective in range(2):indices.update(np.argsort(-scores[:,:,objective].ravel(),kind='stable')[:64].tolist())
  for i in sorted(indices):s,d=divmod(i,64);retained.append(dict(kind='B',dataset='ShareGPT',world=w,batch=32,sample_seed=s,dp_seed=d))
 assert len(retained)<=256;write(PACKET/'retained_candidates.json',retained);manager.drive(retained,'B');winners=select(retained)
 write(ROOT/'search_status.json',dict(status='COMPLETE',winners=len(winners),finished_unix=time.time()));print('SEARCH_COMPLETE',len(winners),flush=True)
if __name__=='__main__':main()
