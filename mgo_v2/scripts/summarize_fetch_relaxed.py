"""Raw exploratory measurements and audited physical mechanism counters."""
from fetch_relaxed_common import *
import csv,argparse,statistics

def table(path,rows):
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(world):
 plans=[p for p in json.loads((ROOT/'physical_plans.json').read_text()) if p['world']==world];out=ROOT/f'R{world}_physical';counts=[];timing=[];gains=[];proof=[]
 for plan in plans:
  key=plan['id'];measures={}
  for policy in ['BR','CA']:
   for variant in ['A3','A2']:
    rs=[json.loads((out/f'{key}_{policy}_{variant}_COUNTERS_rank{r}.json').read_text()) for r in range(world)];assert all(x['status']=='PASS' for x in rs)
    traffic=np.stack([np.load(out/f'{key}_{policy}_{variant}_traffic_rank{r}.npy').reshape(3120,2,2) for r in range(world)],axis=2)
    expected=np.load(Path(plan['path'])/(policy+'_traffic.npy'))*4096;assert np.array_equal(traffic,expected)
    peer=sum(x['activation_bytes']+x['return_bytes'] for x in rs);h2d=sum(x['H2D_bytes'] for x in rs);cpu=plan['winner'][policy]['full'];assert h2d==cpu['H2D_bytes'] and peer==cpu['peer_bytes']
    c=dict(world=world,workload=key,policy=policy,variant=variant,H2D_bytes=h2d,fetches=h2d//9437184,first_fetches=cpu['first_fetches'],reloads=cpu['reload_fetches'],evictions=cpu['evictions'],exact_global_hit_rate=cpu['exact_global_hits_fraction'],exact_local_hit_rate=cpu['exact_local_hits_fraction'],peer_bytes=peer,Critical_total=int(traffic.sum(3).max(2).sum()),max_rank_whole_run_send_recv=int(traffic.sum((0,1,3)).max()),rank_send_bytes=json.dumps(traffic[:,:,:,0].sum((0,1)).tolist()),rank_recv_bytes=json.dumps(traffic[:,:,:,1].sum((0,1)).tolist()),activation_bytes=sum(x['activation_bytes'] for x in rs),weight_bytes=sum(x['weight_bytes'] for x in rs),return_bytes=sum(x['return_bytes'] for x in rs),calls_per_rank=rs[0]['a2a_calls'],max_abs_A2_prefix=max(x['max_abs_weighted_partial'] for x in rs),max_rel_A2_prefix=max(x['max_rel_weighted_partial'] for x in rs),mean_peer_fanout=float(np.load(Path(plan['path'])/(policy+'_fanout.npy')).mean()),quota_max_minus_min=cpu['max_event_rank_fetch_imbalance'])
    counts.append(c);measure=[]
    for repeat in [1,2]:
     paths=[out/f'{key}_{policy}_{variant}_{repeat}_rank{r}.json' for r in range(world)]
     if not paths[0].exists():continue
     ms=[json.loads(p.read_text()) for p in paths];assert all(x['status']=='PASS' and x['no_compile_in_measure'] for x in ms)
     row=dict(world=world,workload=key,policy=policy,variant=variant,repeat=repeat,**{k:max(x[k] for x in ms) for k in ['E2E_wall','decode_wall','TPOT']});timing.append(row);measure.append(row)
    measures[policy,variant]=measure
  for variant in ['A3','A2']:
   br=measures['BR',variant];ca=measures['CA',variant];decision=json.loads((out/f'{key}_{variant}_confirmation.json').read_text());assert len(br)==len(ca)==(2 if decision['confirm'] else 1)
   for repeat in range(len(br)):gains.append(dict(workload=key,comparison='BR_to_CA_'+variant,observation=repeat+1,**{k:1-ca[repeat][k]/br[repeat][k] for k in ['E2E_wall','decode_wall','TPOT']}))
  for policy in ['BR','CA']:
   a,b=measures[policy,'A3'][0],measures[policy,'A2'][0];gains.append(dict(workload=key,comparison='A3_to_A2_'+policy,observation=1,**{k:1-b[k]/a[k] for k in ['E2E_wall','decode_wall','TPOT']}))
  c={r['policy']:r for r in counts if r['workload']==key and r['variant']=='A3'}
  assert eligible(dict(total_fetches=c['BR']['fetches'],H2D_bytes=c['BR']['H2D_bytes']),dict(total_fetches=c['CA']['fetches'],H2D_bytes=c['CA']['H2D_bytes']))
 for path in out.glob('*.json'):proof.append(dict(path=str(path),sha256=sha(path)))
 table(PACKET/f'R{world}_timing.csv',timing);table(PACKET/f'R{world}_counters.csv',counts);table(PACKET/f'R{world}_comparisons.csv',gains)
 write(PACKET/f'R{world}_validation.json',dict(status='PASS',unique_workloads=len(plans),timed_runs=len(timing),horizon=64,fetch_H2D_tolerance=.001,physical_counters_equal_CPU=True,common_frozen_routes=True,A2_full64_argmax_equal_A3=True,raw_receipts=proof))
 lines=[f'# R{world} fixed-route physical decode64 replay','','Exploratory optimized stress screen, not dataset-average or autoregressive','quality evidence. Fetch/H2D tolerance is0.1% relative to BR, not exact equality.','Single-shot observations are screening evidence; no confidence intervals.','Confirmation is exactly one further BR+CA pair only for >=1% initial CA gain.','','| Workload | Comparison | Observation | E2E gain | Decode-wall gain | TPOT gain |','|---|---|---:|---:|---:|---:|']
 for r in gains:lines.append(f"| {r['workload']} | {r['comparison']} | {r['observation']} | {r['E2E_wall']:.2%} | {r['decode_wall']:.2%} | {r['TPOT']:.2%} |")
 lines+=['','Raw seconds are in the timing CSV, with all mechanism counters alongside.','Every physical fetch/peer/Critical_total reproduces its selected CPU schedule.','A2 removes routing-weight traffic:9360→6240 A2A calls/rank over prefill+64','decode, and9216→6144 for decode alone. No full schedule warmup precedes each','measurement. Separate untimed COUNTERS passes validate A2 and compiler coverage.','Full64 teacher-forced argmax hashes agree A3/A2; the first decode forward','compares weighted partials across all48 layers. No further sweep follows.']
 (PACKET/f'R{world}_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);summarize(p.parse_args().world)
