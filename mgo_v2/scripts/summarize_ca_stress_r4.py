"""Summarize the single optimized stress workload with separate physical counters."""
import json,csv,statistics
import numpy as np
from br_carep_cpu import METRICS
from run_ca_stress_r4 import ROOT,PACKET,P,write,sha,samples

def table(name,rows):
 with (PACKET/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 timing=[];summary=[];counts=[];proofs=[]
 candidate=__import__('pathlib').Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/candidates/ShareGPT_R4_B8_s205_d4.json')
 expected=next(r for r in json.loads(candidate.read_text())['results'] if r['cache']==30)
 cpu={'CA':expected['CA']['full'],'BR':next(r['full'] for r in expected['BR'] if r['seed']==73)}
 for env in ['env1','env2']:
  assert json.loads((PACKET/('transport_'+env+'.json')).read_text())['status']=='PASS'
  for policy in ['BR','CA']:
   rows=samples(policy,env);assert len(rows) in [3,5];timing+=rows;r=dict(environment=env,policy=policy,n=len(rows))
   for key in ['E2E_wall','decode_wall','TPOT']:
    values=[x[key] for x in rows];med=statistics.median(values);r.update({key+'_median':med,key+'_min':min(values),key+'_max':max(values),key+'_spread_percent':100*(max(values)-min(values))/med})
   summary.append(r);path=ROOT/f'{policy}_{env}_COUNTERS_0';receipts=[json.loads((path/f'rank{i}.json').read_text()) for i in range(4)]
   assert all(r['status']=='PASS' for r in receipts);logical=np.array([r['logical_counters'] for r in receipts]);assert np.all(logical==logical[0])
   events=np.load(path/'rank0_metrics.npy');c=dict(zip(METRICS,logical[0].tolist()));peer=sum(r['actual_peer_bytes'] for r in receipts);h2d=sum(r['actual_H2D_bytes'] for r in receipts)
   assert peer==(c['remote_token_rank_pairs']+c['remote_expert_routes'])*4096
   assert h2d>0 and h2d==(c['first_fetches']+c['reload_fetches'])*9437184 and c['replica_fetches']==0
   assert events[:,24].max()<=1843 and (events[:,44]-events[:,45]).max()<=1
   counts.append(dict(environment=env,policy=policy,peer_bytes=peer,H2D_bytes=h2d,fetches=int(c['first_fetches']+c['reload_fetches']),reloads=int(c['reload_fetches']),evictions=int(c['evictions']),exact_global_hit_rate=c['exact_global_hits']/c['raw_routes'],exact_local_hit_rate=c['exact_local_hits']/c['raw_routes'],quota_max_minus_min=int((events[:,44]-events[:,45]).max()),peak_resident_copies=int(events[:,24].max()),CPU_expected_peer_bytes=cpu[policy]['peer_bytes'],CPU_expected_H2D_bytes=cpu[policy]['H2D_bytes'],peer_equal_CPU=peer==cpu[policy]['peer_bytes'],H2D_equal_CPU=h2d==cpu[policy]['H2D_bytes']))
 for policy in ['BR','CA']:
  a,b=[r for r in counts if r['policy']==policy]
  for key in ['peer_bytes','H2D_bytes','fetches','reloads','evictions','exact_global_hit_rate','exact_local_hit_rate']:assert a[key]==b[key],(policy,key,'transport changed counters')
 gains=[]
 for env in ['env1','env2']:
  a=next(r for r in summary if r['environment']==env and r['policy']=='BR');b=next(r for r in summary if r['environment']==env and r['policy']=='CA')
  c={r['policy']:r for r in counts if r['environment']==env}
  gains.append(dict(environment=env,**{key+'_gain':1-b[key+'_median']/a[key+'_median'] for key in ['E2E_wall','decode_wall','TPOT']},peer_reduction=1-c['CA']['peer_bytes']/c['BR']['peer_bytes'],H2D_change=c['CA']['H2D_bytes']/c['BR']['H2D_bytes']-1))
 for path in ROOT.glob('*/rank[0-9].json'):proofs.append(dict(path=str(path),sha256=sha(path)))
 stable=all(r[key+'_spread_percent']<=5 for r in summary for key in ['E2E_wall','decode_wall','TPOT'])
 table('timed_samples.csv',timing);table('timing_summary.csv',summary);table('counter_summary.csv',counts);table('comparisons.csv',gains)
 write(PACKET/'validation.json',dict(status='PASS',timing_stable=stable,timed_generations=len(timing),policy_environment_cells=4,raw_receipts=proofs,counters_separate=True,CPU_reference_sha256=sha(candidate),all_peer_equal_CPU=all(r['peer_equal_CPU'] for r in counts),all_H2D_equal_CPU=all(r['H2D_equal_CPU'] for r in counts)))
 lines=['# Physical CA stress validation','','Optimized communication-stress workload, not dataset-average behavior.','ShareGPT R4/local B8/cache30, Gate W128, substitution OFF, decode256,','sample205 / DP4 / BR73. Original eight-BR-seed resource reduction range:','29.6355–30.1124%, median29.8593%.','','Timing stability: '+('PASS' if stable else 'UNSTABLE; do not treat median ordering as established')+'.','','| Env | Policy | n | E2E median [min,max], s | Decode median [min,max], s | TPOT median [min,max], ms |','|---|---|---:|---:|---:|---:|']
 for r in summary:
  vals=[f"{r[k+'_median']*scale:.3f} [{r[k+'_min']*scale:.3f}, {r[k+'_max']*scale:.3f}]" for k,scale in [('E2E_wall',1),('decode_wall',1),('TPOT',1000)]]
  lines.append(f"| {r['environment']} | {r['policy']} | {r['n']} | "+' | '.join(vals)+' |')
 lines+=['','| Env | E2E gain BR→CA | TPOT gain | Physical peer reduction | H2D change |','|---|---:|---:|---:|---:|']
 for r in gains:lines.append(f"| {r['environment']} | {r['E2E_wall_gain']:.2%} | {r['TPOT_gain']:.2%} | {r['peer_reduction']:.2%} | {r['H2D_change']:.2%} |")
 lines+=['',f"Env2 minus Env1 E2E gain: {100*(gains[1]['E2E_wall_gain']-gains[0]['E2E_wall_gain']):.3f} percentage points.",'','Each live PLAN freezes its own physical route/token/cache trajectory. Hashes','must reproduce through COMPILE, all MEASURE repeats and COUNTERS. CPU byte','expectations are compared explicitly in counter_summary.csv; a difference is','not silently treated as an exact match. PLAN_CPU_comparison.json records','route and token differences against the original native capture.','No controller, detailed counters, profiler or compilation runs in MEASURE.','No concurrent resource scan runs between GO and process exit.','No extra workload, policy or timing matrix is launched automatically.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
