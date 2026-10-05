"""Held-out validation with fixed model; no coefficient changes after seeing results."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from critical_stage_b import BROOT,PACKET,POLICIES,read,sha,write

def metric(pred,obs):
 p=np.asarray(pred);y=np.asarray(obs);assert np.all(y>0)
 return dict(spearman=float(spearmanr(p,y).statistic),median_absolute_relative_error=float(np.median(abs(p-y)/y)),predicted_phase_ms_per_step=float(p.sum()/64),observed_phase_ms_per_step=float(y.sum()/64),phase_aggregate_relative_error=float(abs(p.sum()-y.sum())/y.sum()))
def main():
 sources={};model=read(BROOT/'frozen_model.json',sources);assert model==json.loads((PACKET/'CRITICAL_PATH_MODEL.json').read_text())
 validation=[];allrows=[];receipts=[]
 for c in ('C30','C60'):
  for pol in POLICIES:
   path=BROOT/f'{c}_{pol}.json';x=read(path,sources);assert x['status']=='PASS';rows=x['rows'];assert len(rows)==3072
   sources.update(x['source_sha256']);receipts.append(dict(cache=c,policy=pol,raw_features=str(path),sha256=sha(path),seconds=x['seconds'],final_rss_bytes=x['rss_bytes']))
   take=lambda key:[r[key] for r in rows]
   predret=np.array(take('pred_return_rank_ms'));obsret=np.array(take('obs_return_rank_ms'))
   record=dict(cache=c,policy=pol,events=len(rows),forward=metric(take('pred_forward_ms'),take('obs_forward_max_ms')),expert_tau=metric(take('pred_expert_max_ms'),take('obs_expert_max_ms')),expert_rows=metric(take('pred_row_expert_max_ms'),take('obs_expert_max_ms')),return_residency=metric(predret.ravel(),obsret.ravel()),return_max=metric(predret.max(1),obsret.max(1)),critical_tau=metric(take('pred_critical_ms'),take('obs_critical_ms')),critical_rows=metric(take('pred_row_critical_ms'),take('obs_critical_ms')),return_rank_distribution=dict(predicted_p10_p50_p90_ms=np.quantile(predret,[.1,.5,.9]).tolist(),observed_p10_p50_p90_ms=np.quantile(obsret,[.1,.5,.9]).tolist()),h2d_mean_rank_ms_per_step=float(np.array(take('h2d_by_source_event_rank_ms')).sum()/4/64),mandatory_fetches=int(np.array(take('mandatory_fetches_by_rank')).sum()),transport_volume_extrapolation_events=sum(take('transport_volume_out_of_calibration')))
   # Supplemental mechanism-only comparison, keeping the predeclared elapsed
   # target unchanged. Existing interval receipts prove no COMM/COMPUTE overlap.
   from prepare_critical_microbench import OLD
   capture=OLD/c/f'POLICY_REGIME_PROFILE_{c}_NSYS2025_V3_OPT_PF_OVERLAP_{pol}_B128_H64'
   for rank in range(4):
    z=json.loads((capture/f'interval_analysis/rank{rank}_intervals.json').read_text())['interval_decomposition']['aggregate_exclusive_ms']
    assert z['COMM_COMPUTE']==z['H2D_COMM_COMPUTE']==0
   active=[max(sum(r[k][rank] for k in ('obs_forward_rank_ms','obs_expert_rank_ms','obs_return_rank_ms')) for rank in range(4)) for r in rows]
   record['active_comm_expert_tau']=metric(take('pred_critical_ms'),active)
   record['active_scope']='Supplemental max rank-local union of forward/expert/return kernels, after verifying zero phase overlap; excludes gaps, not wall time. Predeclared elapsed target is unchanged.'
   validation.append(record)
   # Portable event CSV/JSON includes scalar and rank predictions/measurements.
   # Detailed owner lists and GPU spans stay in hashed raw sidecars (no giant profiler).
   for r in rows:
    allrows.append({k:v for k,v in r.items() if k not in ('executing_expert_rows_by_rank','current_active_owners','rank_gpu_spans')})
 (PACKET/'MODEL_EVENT_FEATURES.json').write_text(json.dumps(dict(scope='Complete event scores/splits; exact owner/expert lists and absolute GPU timestamps in hashed raw sidecars.',raw_sidecars=receipts,rows=allrows),separators=(',',':'))+'\n')
 scalar_keys=[k for k,v in allrows[0].items() if not isinstance(v,(dict,list))]
 with (PACKET/'MODEL_EVENT_FEATURES.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=scalar_keys,extrasaction='ignore');w.writeheader();w.writerows(allrows)
 order={}
 for c in ('C30','C60'):
  xs={x['policy']:x for x in validation if x['cache']==c}
  for m in ('critical_tau','critical_rows'):
   v={k:x[m]['predicted_phase_ms_per_step'] for k,x in xs.items()};close=abs(v['BR']-v['LA_CA'])/np.mean([v['BR'],v['LA_CA']])<=.02
   order[c+'/'+m]=dict(predicted_ms_per_step=v,pass_gate=bool(close and max(v['BR'],v['LA_CA'])<v['OLD_CA']<v['FCA']))
 numeric={m:all(x[m]['spearman']>=.6 and x[m]['phase_aggregate_relative_error']<=.25 for x in validation) for m in ('critical_tau','critical_rows')}
 tau_better=all(x['expert_tau']['spearman']>=x['expert_rows']['spearman'] and x['expert_tau']['phase_aggregate_relative_error']<=x['expert_rows']['phase_aggregate_relative_error'] for x in validation)
 inflation={}
 for c in ('C30','C60'):
  xs={x['policy']:x for x in validation if x['cache']==c};p=xs['FCA']['return_residency']['predicted_phase_ms_per_step']/xs['BR']['return_residency']['predicted_phase_ms_per_step'];o=xs['FCA']['return_residency']['observed_phase_ms_per_step']/xs['BR']['return_residency']['observed_phase_ms_per_step']
  inflation[c]=dict(predicted_FCA_over_BR=p,observed_FCA_over_BR=o,consistent=bool(p>1 and o>1 and xs['FCA']['return_residency']['spearman']>=.6))
 passed=all(v['pass_gate'] for k,v in order.items() if k.endswith('critical_tau')) and numeric['critical_tau'] and tau_better and all(v['consistent'] for v in inflation.values())
 result=dict(status='PASS' if passed else 'FAIL',stage_c_authorized=False,decision='STOP_FOR_OWNER_REVIEW',calibration_frozen_sha256=sha(BROOT/'frozen_model.json'),rows=validation,ordering=order,numerical_gates_by_model=numeric,tau_not_worse_than_rows=tau_better,arrival_wait_inflation=inflation,phase_aggregate_gate_definition='absolute error of sum over events divided by observed sum; report ms/step. Median event errors also shown, never used to hide aggregate failure.',source_sha256=sources,no_new_gpu_run=True,no_policy_timing_fit=True,no_samples_excluded=True,scope='Env1 eight held-out cells only. Model is GPU-service-only; observed critical span includes actual inter-kernel gaps. NCCL residency includes wait/spin. Neither model score nor sum of causal spans is primary TPOT.')
 write(PACKET/'MODEL_VALIDATION.json',result)
 table=[]
 for x in validation:
  for phase in ('forward','expert_tau','expert_rows','return_residency','return_max','critical_tau','critical_rows','active_comm_expert_tau'):table.append(dict(cache=x['cache'],policy=x['policy'],phase=phase,**x[phase]))
 with (PACKET/'MODEL_VALIDATION.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 lines=['# Stage B: held-out critical-path validation','',f"**{result['status']}: stop before oracle.** Offline Env1 C30/C60 × BR/OLD_CA/FCA/LA_CA, 24,576 decode events. No new GPU inference, no retiming, no exclusions, and no fitting to policy timings.",'','The model was committed before validation at `96f034b`. It uses nonnegative intercept + maximum rank incident packets, fitted only to A1 synthetic shapes separately for forward/return; stable A3 power-ladder tau(n) with linear row interpolation; return base + expert-ready skew. Layer score is forward + maximum expert service + return base, with no duplicated arrival wait.','','## Gates','',f"- Qualitative tau-model ordering, C30/C60: {[order[c+'/critical_tau']['pass_gate'] for c in ('C30','C60')]}.",f'- All-cell critical correlation≥0.60 and phase aggregate error≤25%: {numeric}.',f'- Tau no worse than row-only baseline in both expert correlation and aggregate error: {tau_better}.','', '## Critical score vs measured event span', '', '| Cache | Policy | Pred ms/step | Observed ms/step | Spearman | Aggregate error |', '|---|---|---:|---:|---:|---:|']
 for x in validation:
  m=x['critical_tau'];lines.append(f"| {x['cache']} | {x['policy']} | {m['predicted_phase_ms_per_step']:.3f} | {m['observed_phase_ms_per_step']:.3f} | {m['spearman']:.3f} | {m['phase_aggregate_relative_error']:.1%} |")
 lines+=['', 'Supplemental active-phase-only comparison also fails: maximum rank-local union of forward/expert/return kernels (zero phase overlap verified for all32captures) gives aggregate errors '+', '.join(f"{x['cache']}/{x['policy']} {x['active_comm_expert_tau']['phase_aggregate_relative_error']:.1%}" for x in validation)+'. This removes intervening CPU/H2D-only gaps and does not change the frozen primary target.']
 lines+=['', 'Measured target is the maximum rank-local span from that event’s first forward NCCL kernel start to last return NCCL kernel end. It is read from existing CUPTI timestamps. It includes actual intervening gaps, not a sum of expert max and return max. Summing these event spans does not define TPOT.','', '## Phase and baseline checks', '', '| Cache | Policy | Expert tau corr | Row corr | Expert tau aggregate error | Return corr | H2D mean-rank ms/step |', '|---|---|---:|---:|---:|---:|---:|']
 for x in validation:lines.append(f"| {x['cache']} | {x['policy']} | {x['expert_tau']['spearman']:.3f} | {x['expert_rows']['spearman']:.3f} | {x['expert_tau']['phase_aggregate_relative_error']:.1%} | {x['return_residency']['spearman']:.3f} | {x['h2d_mean_rank_ms_per_step']:.3f} |")
 lines+=['', 'Return residency retains all rank/event values and p10/p50/p90 distributions in JSON. It is not pure wire time. H2D is reported separately and does not enter fitting.','', 'Absolute forward residency aggregate errors are '+f"{min(x['forward']['phase_aggregate_relative_error'] for x in validation):.1%}–{max(x['forward']['phase_aggregate_relative_error'] for x in validation):.1%}"+'; return residency errors are '+f"{min(x['return_residency']['phase_aggregate_relative_error'] for x in validation):.1%}–{max(x['return_residency']['phase_aggregate_relative_error'] for x in validation):.1%}"+'. Matching a policy ratio does not fix this large absolute discrepancy.', '', '## FCA return inflation']
 for c,x in inflation.items():lines.append(f"- {c}: predicted FCA/BR {x['predicted_FCA_over_BR']:.3f}×; observed {x['observed_FCA_over_BR']:.3f}×; consistency gate {x['consistent']}.")
 lines+=['', '## H2D trajectory comparison']
 for cache in ('C30','C60'):
  xs={x['policy']:x for x in validation if x['cache']==cache};br=xs['BR']
  for pol in ('OLD_CA','FCA','LA_CA'):
   x=xs[pol];lines.append(f"- {cache}/{pol} vs BR: observed H2D {x['h2d_mean_rank_ms_per_step']/br['h2d_mean_rank_ms_per_step']-1:+.2%}; mandatory fetches {x['mandatory_fetches']/br['mandatory_fetches']-1:+.2%}. No fitted H2D coefficient.")
 lines+=['', '## Limits and interpretation','', 'A1 establishes isolated concentration sensitivity and A2 establishes isolated skew sensitivity; those facts do not establish that service-time skew suffices in the full runtime. The model omits CPU enqueue gaps, staging/H2D waits, packet packing and routing-weight work. The measured expert attribution contains the full expert loop’s GPU operations, whereas tau measures the compiled expert function span. These are possible missing mechanisms, not newly proven root causes.', '', 'The A1 calibration covers only 1389–1528 packets; extrapolation is explicitly flagged per event. A3 uses the stable primary ladder; the unstable supplemental rank1/n18 point is not used. No coefficients were adjusted to compensate for validation error. Exact per-event split, row, fetch and final controller-state parity with the old captures/proofs must pass before a cell is included.', '', 'Detailed per-event owner/row lists remain in hashed local sidecars. Portable event split/score data and all numerical validation are committed. Ordinal predicted order matches LA_CA < BR < OLD_CA < FCA in both caches; C30 fails only the frozen 2% near-equality convention for LA_CA versus BR. The absolute-time failures independently prevent PASS regardless of that convention. Stage-B failure does not show that all admission optimization is impossible; it rejects this calibrated mechanism as justification for an exact oracle. Per STAGE_B_EXECUTION.md, stop for owner review and do not run Stage C.']
 (PACKET/'STAGE_B_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status=result['status'],ordering=order,numerical=numeric,tau_not_worse=tau_better,inflation=inflation),indent=2))
if __name__=='__main__':main()
