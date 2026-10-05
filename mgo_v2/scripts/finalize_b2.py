"""Publish reproducible B2 conclusion, checks, and shared-resource receipts."""
import csv,json,subprocess,time
from pathlib import Path
import numpy as np
from prepare_critical_microbench import ROOT,PACKET,P,write
from critical_stage_b import sha

def main():
 result=json.loads((PACKET/'B2_PLACEMENT_CAUSALITY.json').read_text());check=json.loads((PACKET/'B2_HOST_COST_PREDICTABILITY.json').read_text());root=ROOT/'b2';state=json.loads((root/'status.json').read_text());assert state['status']=='CAPTURES_COMPLETE' and len(state['completed'])==4
 result['host_cost_predictability_probe']=dict(path=str(PACKET/'B2_HOST_COST_PREDICTABILITY.json'),sha256=sha(PACKET/'B2_HOST_COST_PREDICTABILITY.json'),scope='Diagnostic work-count predictability and exploratory held-out delta rejection, not an accepted revised oracle model.',prefix_checks=check['prefix_heldout'],arrival_delta_checks=check['heldout_host_count_arrival_support'],old_delta_checks=check['old_capture_exploratory_delta'])
 result['classification']=[x for x in result['classification'] if x['term']!='owner-dependent number/shape of compiled host calls and wrappers']
 result['classification'].append(dict(term='owner-dependent number/shape of compiled host calls and wrappers',category='A_CANDIDATE_WITH_B_VARIABILITY',evidence='BR first4steps calibrate cost; last4steps BR/FCA validate rank-event totals within0.05–19.4%, but arrival-delta error is62.3% atC30 and23.8% atC60; old FCA deltas err72.8%/42.7%.',limitation='Ownership controls work count; the coefficient depends on runtime regime/rank/scheduling. A dependable controllable-delta model is not yet validated.'))
 result['candidate_host_cost_probe_run']=True
 for term in result['classification']:
  if term['term']=='host expert wrapper / selection / kernel enqueue / other':term['category']='B_WITH_A_WORK_COUNT_COMPONENT'
 write(PACKET/'B2_PLACEMENT_CAUSALITY.json',result)
 doc=PACKET/'B2_RESULTS.md';text=doc.read_text();marker='\n## Host-work predictability probe\n'
 if marker in text:text=text.split(marker)[0]
 lines=[marker,'The number/shape of expert host calls is owner-dependent, so it is an admission-controllable work-volume candidate. This is stronger than labeling all host work uncontrollable, but weaker than validating a placement cost. No compilation occurred inside captures: the long compiled-host-call range is invocation/enqueue/backpressure of an already compiled function, not compiler time.','', '| Cache | Policy | Held-out rank events | Host-loop Spearman | Aggregate error |', '|---|---|---:|---:|---:|']
 for r in check['prefix_heldout']:lines.append(f"| {r['cache']} | {r['policy']} | {r['heldout_rank_events']} | {r['spearman']:.3f} | {r['aggregate_error']:.1%} |")
 lines+=['', 'Calibration uses only BR steps0–3 and direct per-call measured host medians plus measured noncompiled overhead per expert, separately per cache. BR/FCA steps4–7 are held out; no old TPOT coefficients or arbitrary global scale are fitted. The C30 held-out return-wait delta error is62.3%; C60 is23.8%. This prevents a robust placement-causality pass across both caches.', '', '| Cache | Policy | Predicted delta ms/step | Observed profile delta ms/step | Delta Spearman | Delta error |', '|---|---|---:|---:|---:|---:|']
 for r in check['old_capture_exploratory_delta']:lines.append(f"| {r['cache']} | {r['policy']} | {r['predicted_delta_ms_per_step']:.3f} | {r['observed_delta_ms_per_step']:.3f} | {r['delta_spearman']:.3f} | {r['aggregate_delta_relative_error']:.1%} |")
 lines+=['', 'These old-capture comparisons are exploratory diagnostic rejection checks, not an accepted revised oracle model. Targets are instrumented causal event spans, not primary TPOT. They fail the25% delta-error gate and some correlation/direction checks; no revised model is approved.','', 'Host collective-entry spread and GPU NCCL-start spread correlate0.980–0.994 forward and0.9994–0.9998 return. This supports host-arrival/launch imbalance as the main measured origin of NCCL waiting. It does not identify a unique Python/GIL, driver or OS cause. Slot GPU-wait upper bounds are0–0.534 mean-rank ms/step in these captures; do not equate negligible exposed slot wait with zero CPU staging interference.', '', 'The practical finding is that wire service and expert GPU arithmetic omit the dominant host-side runtime budget. The next cost model would need a validated owner-dependent host-work component and its regime dependence. No runtime/controller optimization, new policy, new LA_CA capture or oracle was launched.']
 doc.write_text(text+'\n'.join(lines)+'\n')
 receipts=[];provenance={}
 for path in state['completed']:
  cap=Path(path);status=json.loads((cap/'status.json').read_text());assert status['status']=='PASS' and status['physical_gpus']==[0,1,4,5]
  receipts.append(dict(capture=path,status=status))
  for rank in range(4):
   assert status['ranks'][rank]['no_compile_in_capture']
   for rel in [f'profile_rank{rank}.nsys-rep',f'interval_analysis/rank{rank}.sqlite',f'b2_rank{rank}_analysis.json',f'copy_trace_rank{rank}.json',f'rank{rank}.json']:
    f=cap/rel;provenance[str(f)]=sha(f)
 write(PACKET/'B2_SOURCE_RECEIPTS.json',dict(captures=receipts,raw_source_sha256=provenance))
 for name in ('B2_FORWARD_ATTRIBUTION','B2_RETURN_ATTRIBUTION','B2_EXPERT_ATTRIBUTION','B2_EVENT_TIMELINE'):
  rows=json.loads((PACKET/(name+'.json')).read_text())['rows'];assert len(rows)==6144
  assert {(r['cache'],r['policy'],r['event'],r['rank']) for r in rows}=={(c,p,e,r) for c in ('C30','C60') for p in ('BR','FCA') for e in range(48,432) for r in range(4)}
  if name=='B2_EVENT_TIMELINE':
   for r in rows:
    a=r['expert_exclusive_account'];assert abs(sum(a[k] for k in ('compiled_gpu_exclusive_ms','ready_wait_exclusive_ms','wrapper_exclusive_ms','loop_other_exclusive_ms','unexplained_ms'))-a['full_loop_span_ms'])<1e-6
  if name in ('B2_FORWARD_ATTRIBUTION','B2_RETURN_ATTRIBUTION'):
   for r in rows:assert abs(r['wire_hat_ms']+r['arrival_wait_hat_ms']+r['residual_ms']-r['nccl_residency_ms'])<1e-9
 assert max(x['p99_anchor_residual_us'] for x in result['clock_alignment'])<50
 ps=json.loads(Path('/home/hwlee/mgo-results/model_inference_load_20261003/processes.json').read_text());assert {x['gpu'] for x in ps}=={0,1,4,5}
 for r in ps:assert 'model_inference_load.py' in Path(f"/proc/{r['pid']}/cmdline").read_bytes().decode()
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True)
 manifest=json.loads((PACKET/'B2_INSTRUMENTATION_MANIFEST.json').read_text());manifest.update(status='COMPLETE',source_code_sha256={str(p):sha(p) for p in [P/'scripts/b2_instrumentation.py',P/'scripts/analyze_b2_rank.py',P/'scripts/report_b2.py',P/'scripts/b2_host_cost_check.py',P/'scripts/finalize_b2.py',P/'examples/refactor_profile_worker.py']});write(PACKET/'B2_INSTRUMENTATION_MANIFEST.json',manifest)
 write(PACKET/'B2_AUDIT.json',dict(status='PASS',unix=time.time(),primary_diagnostic_cells=4,global_events=1536,rank_events=6144,unchanged_token_cache_copy_quota_checks=True,independent_accounting_identities=True,clock_p99_max_us=max(x['p99_anchor_residual_us'] for x in result['clock_alignment']),latency_exclusions=0,failed_attempt_preserved=True,original_BR_supplemental=True,mechanism_gate=result['mechanism_gate'],placement_delta_gate='NOT_PASSED',oracle_run=False,resident_models=ps,gpu_snapshot=gpu,output_sha256={p.name:sha(p) for p in PACKET.glob('B2_*') if p.name!='B2_AUDIT.json'}))
 print('PASS: four captures,6144rank-events, no-double-counting checks, hashes, clock bounds, GPUs0/1/4/5 idle models only')
 print(gpu)
if __name__=='__main__':main()
