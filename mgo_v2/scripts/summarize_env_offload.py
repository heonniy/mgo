"""Validate physical receipts and produce bounded descriptive timing/counters."""
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
from br_carep_cpu import METRICS
from run_env_offload_cell import ROOT,PACKET
CELLS={'P':['BR','CA','CA-rep'],'R':['BR','CA','CA-rep'],'E':['BR','CA']}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csvwrite(name,rows):
 with (PACKET/name).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
 timing=[];counts=[];summaries=[];receipts=[]
 transports=json.loads((PACKET/'transport_preflight.json').read_text());assert len(transports)==4 and all(r['status']=='PASS' for r in transports)
 for cell,policies in CELLS.items():
  for env in ['env1','env2']:
   for policy in policies:
    prefix=f'{cell}_{policy}_{env}';samples=[]
    for path in sorted(ROOT.glob(prefix+'_MEASURE_*')):
     status=json.loads((path/'status.json').read_text());assert status['status']=='PASS'
     rows=[json.loads(p.read_text()) for p in sorted(path.glob('rank*.json'))];world=4 if cell=='R' else 8
     assert len(rows)==world and all(r['status']=='PASS' and r['phase']=='MEASURE' and r['no_compile_in_measure'] for r in rows)
     row=dict(cell=cell,environment=env,policy=policy,repeat=status['repeat'],**{k:max(r[k] for r in rows) for k in ['E2E_wall','TPOT','decode_wall']});timing.append(row);samples.append(row)
     receipts.extend(dict(path=str(p),sha256=sha(p)) for p in path.glob('rank*.json'))
    assert len(samples) in [3,5],(prefix,len(samples))
    summary=dict(cell=cell,environment=env,policy=policy,repeats=len(samples))
    for key in ['E2E_wall','TPOT','decode_wall']:
     values=[r[key] for r in samples];summary.update({key+'_median':statistics.median(values),key+'_min':min(values),key+'_max':max(values)})
    summaries.append(summary)
    path=ROOT/(prefix+'_COUNTERS_0');rows=[json.loads(p.read_text()) for p in sorted(path.glob('rank*.json'))];assert len(rows)==world and all(r['status']=='PASS' and r['phase']=='COUNTERS' for r in rows)
    logical=np.array([r['logical_counters'] for r in rows]);assert np.all(logical==logical[0]);c=dict(zip(METRICS,logical[0].tolist()))
    event_rows=np.load(path/'rank0_metrics.npy');c['peak_resident_copies']=int(event_rows[:,24].max());assert c['peak_resident_copies']<=sum(r['cache_capacity'] for r in rows)
    for key in ['resident_copies','unique_resident_experts','duplicate_copies']:c[key+'_event_sum']=c.pop(key)
    h2d=sum(r['actual_H2D_bytes'] for r in rows);peer=sum(r['actual_peer_bytes'] for r in rows)
    assert h2d>0 and h2d==sum(c[k] for k in ['first_fetches','reload_fetches','replica_fetches'])*9437184
    assert peer==(c['remote_token_rank_pairs']+c['remote_expert_routes'])*4096
    c.update(cell=cell,environment=env,policy=policy,H2D_bytes=h2d,peer_bytes=peer,reload_bytes=c['reload_fetches']*9437184,local_service_fraction=c['local_services']/c['effective_routes'])
    counts.append(c);receipts.extend(dict(path=str(p),sha256=sha(p)) for p in path.glob('rank*.json'))
 gains=[]
 for cell in CELLS:
  pairs=[('BR','CA')] if cell=='E' else [('BR','CA'),('CA','CA-rep')] if cell=='P' else [('CA','CA-rep')]
  for left,right in pairs:
   for env in ['env1','env2']:
    a=next(r for r in summaries if (r['cell'],r['environment'],r['policy'])==(cell,env,left));b=next(r for r in summaries if (r['cell'],r['environment'],r['policy'])==(cell,env,right))
    gains.append(dict(cell=cell,environment=env,comparison=left+' -> '+right,**{k+'_gain':(a[k+'_median']-b[k+'_median'])/a[k+'_median'] for k in ['E2E_wall','TPOT']}))
 csvwrite('timed_samples.csv',timing);csvwrite('timing_summary.csv',summaries);csvwrite('counter_summary.csv',counts);csvwrite('comparisons.csv',gains)
 (PACKET/'validation.json').write_text(json.dumps(dict(status='PASS',policy_environment_cells=len(summaries),timed_generations=len(timing),counters_separate=True,transfer_bytes_verified=True,raw_receipts=receipts),indent=2)+'\n')
 lines=['# Physical Env E2E / TPOT results','','Real Qwen3 expert offloading with CPU-resident weights, bounded GPU cache,','actual miss H2D, inter-rank activations and GPU expert execution. Controller','planning, compilation and counters are outside timing. No quality or physical','PCIe-equivalence claim.','','| Cell | Env | Policy | n | E2E median [range], s | TPOT median [range], ms |','|---|---|---|---:|---:|---:|']
 for r in summaries:lines.append(f"| {r['cell']} | {r['environment']} | {r['policy']} | {r['repeats']} | {r['E2E_wall_median']:.3f} [{r['E2E_wall_min']:.3f}, {r['E2E_wall_max']:.3f}] | {1000*r['TPOT_median']:.3f} [{1000*r['TPOT_min']:.3f}, {1000*r['TPOT_max']:.3f}] |")
 lines+=['','## Comparisons','','Positive gain means the right policy is faster. These are median ratios.','','| Cell | Env | Comparison | E2E gain | TPOT gain |','|---|---|---|---:|---:|']
 for r in gains:lines.append(f"| {r['cell']} | {r['environment']} | {r['comparison']} | {r['E2E_wall_gain']:.2%} | {r['TPOT_gain']:.2%} |")
 lines+=['','Separate counters are in `counter_summary.csv`; raw sample ranges are in','`timed_samples.csv`. See EXECUTION.md for the frozen-baseline demand oracle,','pageable CPU source transfers, generation convention and resource corrections.','Policy-specific substitution trajectories are validated against their own PLAN,','not forced to the exact-only master trace. Timing alone does not establish a','communication mechanism; inspect peer/H2D/reload changes alongside each pair.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
