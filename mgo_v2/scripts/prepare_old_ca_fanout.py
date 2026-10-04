"""Bounded CPU preparation; reuse all BR/CA receipts and replay OldCA only."""
from fetch_matched_common import *
import resource,time
import run_ca_stress_cpu as manager
from fetch_matched_replay import replay_critical
from run_br_cpu import summarize
OLD_ROOT=ROOT
ROOT=Path('/home/hwlee/mgo-results/old_ca_fanout_followup_20261004')
PACKET=P/'experiments/old_ca_fanout_followup_20261004'

def within(br,other,denominator):
 return all(abs(br['full'][k]-other['full'][k])*denominator<=br['full'][k] for k in ('total_fetches','H2D_bytes'))

def execute(a,world):
 cap=np.array([1843//world+(r<1843%world) for r in range(world)],np.int64)
 result=replay_critical(a['selected'],a['weights'],a['offsets'],a['prefill_origins'],a['decode_origins'],a['gates'],np.zeros((48,128,128),np.float32),cap,64,True,False,3,42)
 rows,fetch=result[:2];traffic=result[-1];full=summarize(rows,fetch,1843,64);h=hashlib.sha256()
 for arr in result[2:-1]:h.update(arr.tobytes())
 assert int(traffic[:,:,:,0].sum())*4096==full['peer_bytes']
 assert np.array_equal(traffic[:,:,:,0].sum(2),traffic[:,:,:,1].sum(2))
 assert full['max_event_rank_fetch_imbalance']<=1 and full['peak_resident_copies']<=1843
 assert full['replica_fetches']==0
 fanout=traffic[:,0,:,0].sum(1)
 assert int(fanout.sum())==int(rows[:,30].sum())
 return dict(policy='OldCA-fanout',seed=42,full=full,decode=summarize(rows[48:],fetch[48:],1843,64),Fanout_total=int(fanout.sum()),fanout_event_percentiles=dict(zip(('p50','p95','max'),np.percentile(fanout,[50,95,100]).tolist())),mean_remote_destinations_per_token=float(fanout.sum()/a['offsets'][-1]),Critical_total=int(traffic.sum(3).max(2).sum())*4096,rank_send_bytes=(traffic[:,:,:,0].sum((0,1))*4096).tolist(),rank_recv_bytes=(traffic[:,:,:,1].sum((0,1))*4096).tolist(),dispatch_bytes=int(traffic[:,0,:,0].sum())*4096,return_bytes=int(traffic[:,1,:,0].sum())*4096,final_state_sha256=h.hexdigest())

def worker(tasks,results,cpu):
 os.sched_setaffinity(0,{cpu});resource.setrlimit(resource.RLIMIT_AS,(96*2**30,96*2**30));resource.setrlimit(resource.RLIMIT_DATA,(8*2**30,8*2**30));pool=PrefixPool('ShareGPT')
 sources={name:sha(P/'scripts'/name) for name in ('old_ca_fanout_policy.py','fetch_matched_replay.py','prepare_old_ca_fanout.py')}
 while True:
  job=tasks.get()
  if job is None:return
  start=time.monotonic()
  try:
   source=Path(job['path']);d=json.loads(source.read_text());out=ROOT/'candidates'/source.name
   if out.exists():
    prior=json.loads(out.read_text());assert prior['status']=='PASS' and prior['sources']==sources and prior['candidate_sha256']==sha(source)
   else:
    a=pool.pack(np.array(d['request_ids']),job['world'],32)
    assert hashlib.sha256(a['selected'].tobytes()).hexdigest()==d['route_sha256']
    assert hashlib.sha256(a['gates'].tobytes()).hexdigest()==d['gate_sha256']
    old=execute(a,job['world'])
    write(out,dict(status='PASS',candidate_path=str(source),candidate_sha256=sha(source),sources=sources,OldCA=old,seconds=time.monotonic()-start))
   results.put(dict(status='PASS',job=job,seconds=time.monotonic()-start))
  except BaseException as exc:results.put(dict(status='FAIL',job=job,error=repr(exc)));return

def select(jobs):
 winners=[];diagnostics=[]
 for world in (4,8):
  rows=[]
  for job in jobs:
   if job['world']!=world:continue
   d=json.loads(Path(job['path']).read_text());old=json.loads((ROOT/'candidates'/Path(job['path']).name).read_text())['OldCA'];ca=d['CA']
   for br in d['BR']:
    mismatch=max(abs(x['full']['H2D_bytes']-br['full']['H2D_bytes'])/br['full']['H2D_bytes'] for x in (ca,old))
    row=dict(world=world,sample_seed=d['job']['sample_seed'],dp_seed=d['job']['dp_seed'],BR_seed=br['seed'],BR=br,CA=ca,OldCA=old,request_ids=d['request_ids'],route_sha256=d['route_sha256'],gate_sha256=d['gate_sha256'],candidate_path=job['path'],candidate_sha256=sha(Path(job['path'])),max_H2D_difference_relative=mismatch,fanout_gain=br['full']['remote_token_rank_pairs']-old['Fanout_total'],critical_gain=br['Critical_total']-old['Critical_total'],peer_gain=br['full']['peer_bytes']-old['full']['peer_bytes'])
    row['preferred']=all(within(br,x,1000) for x in (ca,old));row['fallback']=all(within(br,x,400) for x in (ca,old));rows.append(row)
  preferred=[r for r in rows if r['preferred']];fallback=[r for r in rows if r['fallback']]
  def objective(r):return (-r['fanout_gain'],-r['critical_gain'],-r['peer_gain'],r['max_H2D_difference_relative'],r['sample_seed'],r['dp_seed'],r['BR_seed'])
  if preferred:chosen=min(preferred,key=objective);chosen['match']='NEAR_MATCH_010'
  elif fallback:chosen=min(fallback,key=lambda r:(r['max_H2D_difference_relative'],*objective(r)));chosen['match']='NEAR_MATCH_025'
  else:chosen=None
  if chosen:winners.append(chosen)
  diagnostics.append(dict(world=world,eligible_010=len(preferred),eligible_025=len(fallback),status='SELECTED' if chosen else 'SKIP_PHYSICAL',closest=min(rows,key=lambda r:r['max_H2D_difference_relative']) if rows else None))
 write(PACKET/'winners.json',winners);write(PACKET/'selection_validation.json',dict(status='PASS',new_trace_capture=False,new_BR_CA_replays=0,old_ca_replays=len(jobs),diagnostics=diagnostics));return winners

def main():
 ROOT.mkdir(exist_ok=True)
 assert json.loads((PACKET/'policy_validation.json').read_text())['status']=='PASS'
 predecessor=json.loads((P/'experiments/fetch_matched_b32_a2a_20261004/tolerance_001/status.json').read_text());assert predecessor['status']=='COMPLETE'
 jobs=[];pruned=[]
 for source in sorted((OLD_ROOT/'B').glob('*.json')):
  d=json.loads(source.read_text());assert d['status']=='PASS'
  # A common match is impossible if Current-CA already fails the maximum bound.
  if not any(within(br,d['CA'],400) for br in d['BR']):pruned.append(str(source));continue
  jobs.append(dict(path=str(source),world=d['job']['world'],batch=32))
 write(PACKET/'candidate_manifest.json',dict(jobs=jobs,pruned_Current_CA_outside_025=pruned,new_candidate_seeds=0,predecessor_status='COMPLETE'))
 manager.worker=worker;manager.ROOT=ROOT;manager.PACKET=PACKET;manager.drive(jobs,'OldCA')
 winners=select(jobs)
 write(ROOT/'status.json',dict(status='CPU_SELECTION_COMPLETE',physical_started=False,selected_worlds=[x['world'] for x in winners],finished_unix=time.time()))
 print('CPU_SELECTION_COMPLETE',len(winners),flush=True)
if __name__=='__main__':main()
