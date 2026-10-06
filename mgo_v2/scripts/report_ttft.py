"""Report best-seed TTFT estimates separately from selection and diagnostics."""
from ttft_common import *
import statistics

def main():
 state=json.loads((ROOT/'physical_status.json').read_text());assert state['status']=='PASS'
 rows=[]
 for spec in state['selected']:
  out=ROOT/'physical'/('TTFT_S2_'+spec['key']);r=json.loads((out/(spec['key']+'_result.json')).read_text())
  if r['status']!='PASS':continue
  candidate=spec['policy'];by_policy={};copy_totals={};barriers={};returns={};skews={}
  for policy in ('BR',candidate):
   ranks=[json.loads((out/f'{spec["key"]}_{policy}_diagnostic_rank{k}.json').read_text()) for k in range(4)]
   assert all(x['status']=='PASS' and len(x['phases'])==48 and x['validation']['observed']['barriers']==48 for x in ranks)
   copy_totals[policy]=sum(x['validation']['observed']['copies'] for x in ranks)
   barriers[policy]=statistics.mean(sum(t['barrier_ms'] for t in x['phases']) for x in ranks)
   returns[policy]=statistics.mean(sum(t['return_cuda_ms'] for t in x['phases']) for x in ranks)
   skews[policy]=sum((max(x['phases'][l]['rank_ready_ns'] for x in ranks)-min(x['phases'][l]['rank_ready_ns'] for x in ranks))/1e6 for l in range(48))
   by_policy[policy]=dict(samples_s=r['samples'][policy],estimate_s=r['estimates'][policy],range_s=[min(r['samples'][policy]),max(r['samples'][policy])],barrier_mean_rank_ms=barriers[policy],return_phase_mean_rank_cuda_ms=returns[policy],summed_rank_ready_skew_ms=skews[policy],idle_barrier_reference_ms=statistics.mean(x['idle_barrier_reference_ms'] for x in ranks),mandatory_H2D_copies_by_rank=[x['validation']['observed']['copies'] for x in ranks],mandatory_H2D_bytes_by_rank=[x['validation']['observed']['bytes'] for x in ranks],prefill_boundary_state_hashes=[x['validation']['observed']['state_hash'] for x in ranks],wire_bytes_by_rank=[x['validation']['wire_bytes'] for x in ranks],expert_rows_by_rank=[sum(x['validation']['expert_rows']) for x in ranks])
  gain=r['gain'];classification='STRICT_PLACEMENT' if len(set(copy_totals.values()))==1 else 'PLACEMENT_STATE_EFFECT'
  decision='UNSTABLE' if r['unstable'] else 'GO' if gain>=.05 else 'MARGINAL' if gain>=.02 else 'STOP_PREFILL'
  rows.append(dict(cache=spec['cache'],batch=spec['batch'],context=512,phase='prefill',candidate=candidate,workload_seed=spec['workload_seed'],placement_seed=spec['placement_seed'],BR_TTFT_s=r['estimates']['BR'],candidate_TTFT_s=r['estimates'][candidate],gain=gain,classification=classification,unstable=r['unstable'],decision=decision,policies=by_policy,candidate_barrier_share=barriers[candidate]/(1000*r['estimates'][candidate]),candidate_return_phase_share=returns[candidate]/(1000*r['estimates'][candidate])))
 result=dict(status='COMPLETE' if len(rows)==12 else 'INCOMPLETE_VALID_PAIRS',dataset='ShareGPT_LONG512',rows=rows,invalid_regimes=state.get('invalid_regimes',[]),interpretation='BEST-SEED HEADROOM / ORACLE SCREEN. Candidate-selected favorable workload and paired BR admission seed; not average-case performance. Cross-policy token mismatch is INVALID. Diagnostics are separate passes; rank-local phase durations are not an additive global critical path. Idle barrier includes scheduling and cannot be subtracted as exact imbalance.',scope='TTFT/prefill only; decode64 and joint phase policy matrix were not run.')
 write(PACKET/'TTFT_RESULTS.json',result)
 lines=['# ShareGPT_LONG512 TTFT best-seed headroom','','Primary: exact512 valid tokens, R4 GPUs0/1/4/5, BF16, cold expert cache, no substitution or replication. Timing begins with a resident model and ends when the first token is ready; load/compile are excluded.','','**BEST-SEED HEADROOM / ORACLE SCREEN**, not an unbiased average-case comparison. Every gain pairs BR and candidate on identical requests and seed. S1 selection samples are excluded from the fresh-process S2 estimate.','','| Cache | Local batch | Candidate | Workload seed | Placement seed | BR TTFT (s) | Candidate TTFT (s) | Gain | Class | Decision |','|---|---:|---|---:|---:|---:|---:|---:|---|---|']
 for x in rows:lines.append(f"| C{x['cache']} | {x['batch']} | {x['candidate']} | {x['workload_seed']} | {x['placement_seed']} | {x['BR_TTFT_s']:.6f} | {x['candidate_TTFT_s']:.6f} | {100*x['gain']:.3f}% | {x['classification']} | {x['decision']} |")
 lines+=['','All valid repeat samples, full ranges, copy/byte distributions, boundary hashes and separate phase diagnostics are in TTFT_RESULTS.json. Two repeats stop at<=2%; one third is allowed only at2–5%; >5% is unstable without further repeats. No latency-based sample deletion.','','No decode-policy winner or global phase-policy claim is supported by this TTFT-only packet. Invalid regimes (if any): '+json.dumps(state.get('invalid_regimes',[])), '']
 (PACKET/'TTFT_RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
