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
    for key in ['exact_global_hits','exact_local_hits','resident_substitute_hits','effective_hits','residual_miss_routes']:
     c[key+'_fraction_of_raw_routes']=c[key]/c['raw_routes']
    counts.append(c);receipts.extend(dict(path=str(p),sha256=sha(p)) for p in path.glob('rank*.json'))
 gains=[]
 for cell in CELLS:
  pairs=[('BR','CA')] if cell=='E' else [('BR','CA'),('CA','CA-rep')] if cell=='P' else [('CA','CA-rep')]
  for left,right in pairs:
   for env in ['env1','env2']:
    a=next(r for r in summaries if (r['cell'],r['environment'],r['policy'])==(cell,env,left));b=next(r for r in summaries if (r['cell'],r['environment'],r['policy'])==(cell,env,right))
    gains.append(dict(cell=cell,environment=env,comparison=left+' -> '+right,**{k+'_gain':(a[k+'_median']-b[k+'_median'])/a[k+'_median'] for k in ['E2E_wall','TPOT']}))
 expected=json.loads((PACKET/'expected_cpu_mechanism.json').read_text())['rows'];mechanisms=[];contrasts=[]
 for gain in gains:
  cell= gain['cell'];env=gain['environment'];left,right=gain['comparison'].split(' -> ')
  a=next(r for r in counts if (r['cell'],r['environment'],r['policy'])==(cell,env,left));b=next(r for r in counts if (r['cell'],r['environment'],r['policy'])==(cell,env,right))
  ca=next(r['counters'] for r in expected if (r['cell'],r['policy'])==(cell,left));cb=next(r['counters'] for r in expected if (r['cell'],r['policy'])==(cell,right))
  mechanisms.append(dict(cell=cell,environment=env,comparison=gain['comparison'],expected_CPU_peer_gain=(ca['peer_bytes']-cb['peer_bytes'])/ca['peer_bytes'],physical_peer_gain=(a['peer_bytes']-b['peer_bytes'])/a['peer_bytes'],peer_direction_agrees=np.sign(ca['peer_bytes']-cb['peer_bytes'])==np.sign(a['peer_bytes']-b['peer_bytes']),physical_H2D_change=(b['H2D_bytes']-a['H2D_bytes'])/a['H2D_bytes'],physical_reload_change=(b['reload_bytes']-a['reload_bytes'])/max(1,a['reload_bytes'])))
 for cell in CELLS:
  for policy in CELLS[cell]:
   a=next(r for r in counts if (r['cell'],r['environment'],r['policy'])==(cell,'env1',policy));b=next(r for r in counts if (r['cell'],r['environment'],r['policy'])==(cell,'env2',policy))
   for k in ['H2D_bytes','peer_bytes','reload_bytes','exact_global_hits','exact_local_hits','resident_substitute_hits','effective_hits','residual_miss_routes','replica_fetches','replicas_reused']:assert a[k]==b[k],(cell,policy,'Env counter drift',k)
 for cell in ['P','E']:
  a=next(r for r in gains if r['cell']==cell and r['environment']=='env1' and r['comparison']=='BR -> CA');b=next(r for r in gains if r['cell']==cell and r['environment']=='env2' and r['comparison']=='BR -> CA')
  contrasts.append(dict(cell=cell,**{k+'_Env2_minus_Env1_gain':b[k+'_gain']-a[k+'_gain'] for k in ['E2E_wall','TPOT']}))
 csvwrite('mechanism_checks.csv',mechanisms);csvwrite('environment_contrast.csv',contrasts)
 csvwrite('timed_samples.csv',timing);csvwrite('timing_summary.csv',summaries);csvwrite('counter_summary.csv',counts);csvwrite('comparisons.csv',gains)
 (PACKET/'validation.json').write_text(json.dumps(dict(status='PASS',policy_environment_cells=len(summaries),timed_generations=len(timing),counters_separate=True,transfer_bytes_verified=True,raw_receipts=receipts),indent=2)+'\n')
 lines=['# Physical Env E2E / TPOT results','','Real Qwen3 expert offloading with CPU-resident weights, bounded GPU cache,','actual miss H2D, inter-rank activations and GPU expert execution. Controller','planning, compilation and counters are outside timing. Intervals are the maximum',
'over ranks. No quality or physical','PCIe-equivalence claim.','','| Cell | Env | Policy | n | E2E median [range], s | TPOT median [range], ms |','|---|---|---|---:|---:|---:|']
 for r in summaries:lines.append(f"| {r['cell']} | {r['environment']} | {r['policy']} | {r['repeats']} | {r['E2E_wall_median']:.3f} [{r['E2E_wall_min']:.3f}, {r['E2E_wall_max']:.3f}] | {1000*r['TPOT_median']:.3f} [{1000*r['TPOT_min']:.3f}, {1000*r['TPOT_max']:.3f}] |")
 lines+=['','## Comparisons','','Positive gain means the right policy is faster. These are median ratios.','','| Cell | Env | Comparison | E2E gain | TPOT gain |','|---|---|---|---:|---:|']
 for r in gains:lines.append(f"| {r['cell']} | {r['environment']} | {r['comparison']} | {r['E2E_wall_gain']:.2%} | {r['TPOT_gain']:.2%} |")
 lines+=['','Separate counters are in `counter_summary.csv`; raw sample ranges are in','`timed_samples.csv`. CPU/physical mechanism directions and H2D/reload changes',
'are in `mechanism_checks.csv`; the primary Env contrast is in',
'`environment_contrast.csv`. See EXECUTION.md for the frozen-baseline demand oracle,','pageable CPU source transfers, generation convention and resource corrections.','Policy-specific substitution trajectories are validated against their own PLAN,','not forced to the exact-only master trace. Timing alone does not establish a','communication mechanism; inspect peer/H2D/reload changes alongside each pair.']
 lines+=['','## Decode wall consistency','','| Cell | Env | Policy | Decode wall median [range], s |','|---|---|---|---:|']
 for r in summaries:lines.append(f"| {r['cell']} | {r['environment']} | {r['policy']} | {r['decode_wall_median']:.3f} [{r['decode_wall_min']:.3f}, {r['decode_wall_max']:.3f}] |")
 lines+=['','## Separate counters','','Fractions use raw routed selections as denominator and are not additive.','Volumes are totals over all ranks.','','| Cell | Env | Policy | Exact global hit | Effective hit | Residual miss | H2D GiB | Peer GiB | Reload GiB |','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for r in counts:lines.append(f"| {r['cell']} | {r['environment']} | {r['policy']} | {r['exact_global_hits_fraction_of_raw_routes']:.2%} | {r['effective_hits_fraction_of_raw_routes']:.2%} | {r['residual_miss_routes_fraction_of_raw_routes']:.2%} | {r['H2D_bytes']/2**30:.3f} | {r['peer_bytes']/2**30:.3f} | {r['reload_bytes']/2**30:.3f} |")
 substitution=[]
 for env in ['env1','env2']:
  for policy in ['BR','CA']:
   on=next(r for r in counts if (r['cell'],r['environment'],r['policy'])==('P',env,policy));off=next(r for r in counts if (r['cell'],r['environment'],r['policy'])==('E',env,policy))
   a=next(r for r in summaries if (r['cell'],r['environment'],r['policy'])==('E',env,policy));b=next(r for r in summaries if (r['cell'],r['environment'],r['policy'])==('P',env,policy))
   substitution.append(dict(environment=env,policy=policy,H2D_reduction=(off['H2D_bytes']-on['H2D_bytes'])/off['H2D_bytes'],residual_miss_fraction_off=off['residual_miss_routes_fraction_of_raw_routes'],residual_miss_fraction_on=on['residual_miss_routes_fraction_of_raw_routes'],**{k+'_gain_with_substitution':(a[k+'_median']-b[k+'_median'])/a[k+'_median'] for k in ['E2E_wall','TPOT']}))
 csvwrite('substitution_control.csv',substitution)
 lines+=['','## Substitution control','','P (ON) versus E (OFF) uses identical requests but policy-specific generated','trajectories. This is not a fixed-route causal estimate or an accuracy evaluation.','','| Env | Policy | H2D reduction | Miss fraction OFF → ON | E2E gain ON | TPOT gain ON |','|---|---|---:|---:|---:|---:|']
 for r in substitution:lines.append(f"| {r['environment']} | {r['policy']} | {r['H2D_reduction']:.2%} | {r['residual_miss_fraction_off']:.2%} → {r['residual_miss_fraction_on']:.2%} | {r['E2E_wall_gain_with_substitution']:.2%} | {r['TPOT_gain_with_substitution']:.2%} |")
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
