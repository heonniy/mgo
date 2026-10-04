"""Validate physical receipts against CPU schedules and report bounded screen."""
from prepare_old_ca_fanout import *
import argparse,csv

def table(path,rows):
 with path.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def summarize(world):
 plans=[p for p in json.loads((ROOT/'physical_plans.json').read_text()) if p['world']==world]
 out=ROOT/f'R{world}_physical';counts=[];timings=[];gains=[]
 for plan in plans:
  key=plan['id'];measures={};local={}
  for policy in ['BR','CA','OldCA']:
   rs=[json.loads((out/f'{key}_{policy}_A3_COUNTERS_rank{r}.json').read_text()) for r in range(world)]
   assert all(x['status']=='PASS' and x['a2a_calls']==9360 for x in rs)
   traffic=np.stack([np.load(out/f'{key}_{policy}_A3_traffic_rank{r}.npy').reshape(3120,2,2) for r in range(world)],axis=2)
   expected=np.load(Path(plan['path'])/(policy+'_traffic.npy'))*4096;assert np.array_equal(traffic,expected)
   cpu=plan['winner'][policy];full=cpu['full'];h2d=sum(x['H2D_bytes'] for x in rs);peer=sum(x['activation_bytes']+x['return_bytes'] for x in rs)
   assert h2d==full['H2D_bytes'] and peer==full['peer_bytes']
   critical=int(traffic.sum(3).max(2).sum());assert critical==cpu['Critical_total']
   fanout=traffic[:,0,:,0].sum(1)//4096;assert int(fanout.sum())==full['remote_token_rank_pairs']
   rows=np.load(Path(plan['path'])/(policy+'_metrics.npy'));tokens=rows[:,0].sum()/8
   c=dict(world=world,policy=policy,workload=key,H2D_bytes=h2d,fetches=h2d//9437184,reloads=full['reload_fetches'],peer_bytes=peer,Fanout_total=int(fanout.sum()),mean_remote_destinations_per_token=float(fanout.sum()/tokens),fanout_event_p50=float(np.percentile(fanout,50)),fanout_event_p95=float(np.percentile(fanout,95)),fanout_event_max=int(fanout.max()),Critical_total=critical,activation_bytes=sum(x['activation_bytes'] for x in rs),weight_bytes=sum(x['weight_bytes'] for x in rs),return_bytes=sum(x['return_bytes'] for x in rs),rank_send_bytes=json.dumps(traffic[:,:,:,0].sum((0,1)).tolist()),rank_recv_bytes=json.dumps(traffic[:,:,:,1].sum((0,1)).tolist()),exact_global_hit_rate=full['exact_global_hits_fraction'],exact_local_hit_rate=full['exact_local_hits_fraction'],evictions=full['evictions'])
   counts.append(c);local[policy]=c
   for repeat in (1,2,3):
    paths=[out/f'{key}_{policy}_A3_{repeat}_rank{r}.json' for r in range(world)]
    if not paths[0].exists():continue
    receipts=[json.loads(path.read_text()) for path in paths];assert all(x['status']=='PASS' and x['no_compile_in_measure'] for x in receipts)
    for rank,x in enumerate(receipts):
     assert x['schedule_sha256']==plan['proofs'][policy][rank]['schedule_sha256']
     assert x['argmax_hash']==rs[rank]['argmax_hash'] and x['teacher_hash']==rs[rank]['teacher_hash']
    row=dict(world=world,workload=key,policy=policy,repeat=repeat,**{k:max(x[k] for x in receipts) for k in ['E2E_wall','decode_wall','TPOT']})
    timings.append(row);measures[policy,repeat]=row
  for policy in ['CA','OldCA']:
   denom=1000 if plan['winner']['match']=='NEAR_MATCH_010' else 400
   assert within(plan['winner']['BR'],plan['winner'][policy],denom)
   decision=json.loads((out/f'{key}_{policy}_confirmation.json').read_text())
   initial={k:1-measures[policy,1][k]/measures['BR',1][k] for k in ['E2E_wall','decode_wall','TPOT']}
   assert decision['confirm']==(max(initial[k] for k in ['decode_wall','TPOT'])>=.01)
   pairs=[(1,1)]+([(decision['BR_repeat'],2)] if decision['confirm'] else [])
   for index,(br_rep,ca_rep) in enumerate(pairs,1):
    gains.append(dict(world=world,workload=key,policy=policy,observation=index,BR_repeat=br_rep,policy_repeat=ca_rep,H2D_delta=local[policy]['H2D_bytes']/local['BR']['H2D_bytes']-1,peer_reduction=1-local[policy]['peer_bytes']/local['BR']['peer_bytes'],fanout_reduction=1-local[policy]['Fanout_total']/local['BR']['Fanout_total'],critical_reduction=1-local[policy]['Critical_total']/local['BR']['Critical_total'],**{k:1-measures[policy,ca_rep][k]/measures['BR',br_rep][k] for k in initial}))
  expected_runs=3+sum(2 for policy in ['CA','OldCA'] if json.loads((out/f'{key}_{policy}_confirmation.json').read_text())['confirm'])
  assert len(measures)==expected_runs
 table(PACKET/f'R{world}_timing.csv',timings);table(PACKET/f'R{world}_counters.csv',counts);table(PACKET/f'R{world}_comparisons.csv',gains)
 write(PACKET/f'R{world}_validation.json',dict(status='PASS',runtime='A3',horizon=64,timed_runs=len(timings),physical_counters_equal_CPU=True,new_trace_capture=False,raw_receipts=[dict(path=str(p),sha256=sha(p)) for p in sorted(out.glob('*.json'))]))
 lines=[f'# R{world} OldCA fan-out screen','','Exploratory, selected-workload single-shot screen. Positive timing gains mean faster than BR.','Confirmation observations are separate; no stability or dataset-average claim.','','| Policy | Observation | H2D delta | Peer reduction | Fan-out reduction | Critical reduction | TPOT gain | Decode-wall gain |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in gains:lines.append(f"| {r['policy']} | {r['observation']} | {r['H2D_delta']:.4%} | {r['peer_reduction']:.2%} | {r['fanout_reduction']:.2%} | {r['critical_reduction']:.2%} | {r['TPOT']:.2%} | {r['decode_wall']:.2%} |")
 lines+=['','See CSV files for raw seconds, E2E and all counters. Physical communication and H2D match audited CPU schedules.','No same/path affinity, A2, Env1, replication or decode256 extension.']
 (PACKET/f'R{world}_RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--world',type=int,required=True);summarize(parser.parse_args().world)
