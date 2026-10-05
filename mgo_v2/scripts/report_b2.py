"""B2 measured arrival/queue accounting, without a placement-causality shortcut."""
import csv,json,subprocess,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from prepare_critical_microbench import P,PACKET,ROOT,write
from critical_stage_b import sha
from analyze_b2_rank import analyze
B2=ROOT/'b2'
NSYS='/home/hwlee/mgo-tools/nsight-2025.3/extracted/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys'
def export(capture):
 out=capture/'interval_analysis'
 if not (out/'summary.json').exists():
  assert not out.exists(),'preserve partial export for explicit repair'
  with (capture/'export_driver.log').open('w') as log:subprocess.run(['/home/hwlee/sub-moe/phase01/.venv/bin/python',str(P/'scripts/export_refactor_profiles.py'),str(capture),'--nsys',NSYS],stdout=log,stderr=subprocess.STDOUT,check=True)
 result=[]
 for rank in range(4):
  f=capture/f'b2_rank{rank}_analysis.json'
  if f.exists():x=json.loads(f.read_text())
  else:x=analyze(capture,rank);write(f,x)
  result.append(x)
 return result

def report():
 state=json.loads((B2/'status.json').read_text());assert state['status']=='CAPTURES_COMPLETE'
 model=json.loads((PACKET/'CRITICAL_PATH_MODEL.json').read_text());forward=[];back=[];expert=[];timeline=[];sources={};clocks=[];cells=[]
 for path in state['completed']:
  cap=Path(path);cache=cap.parent.name;policy=json.loads((cap/'case.json').read_text())['policy'];ranks=export(cap)
  comm=[json.loads((cap/f'communication_rank{r}.json').read_text())['events'] for r in range(4)]
  original=json.loads((ROOT/'stage_b'/f'{cache}_{policy}.json').read_text())['rows'][:384]
  for rank in range(4):
   clocks.append(dict(cache=cache,policy=policy,rank=rank,**ranks[rank]['clock_alignment']))
   for name in (f'b2_host_rank{rank}.json',f'b2_rank{rank}_analysis.json',f'communication_rank{rank}.json',f'rank{rank}.json'):
    sources[str(cap/name)]=sha(cap/name)
  for i in range(384):
   event=i+48;rs=[x['events'][i] for x in ranks];assert all(x['event']==event for x in rs)
   S=np.array([x[i]['send_token_rows'] for x in comm]);np.fill_diagonal(S,0);old=original[i];assert S.tolist()==old['split'],'new prefix changed traffic'
   for rank,r in enumerate(rs):
    calls=r['expert_calls'];assert sorted((x['expert'],x['rows']) for x in calls)==sorted(map(tuple,old['executing_expert_rows_by_rank'][rank])),'expert row mismatch'
    phases=r['phases'];loop=phases['expert_loop'];compiled=phases['expert_compiled_kernel'];ready=phases.get('expert_ready_wait',{}).get('host_union_ms',0.)
    gather=phases['expert_gather'];weight=phases['expert_weight_partial'];record=phases['expert_record_use']
    tau=float(sum(np.interp(x['rows'],model['tau']['rows'],model['tau']['ms']) for x in calls))
    # Exact host-loop partition. Host kernel-call time includes enqueue and
    # occasional backpressure; it is distinct from actual kernel service.
    other=loop['host_other_ms'];select=loop['host_ready_select_without_wait_ms']
    hostparts=ready+select+gather['host_union_ms']+compiled['host_union_ms']+record['host_union_ms']+weight['host_union_ms']+other
    assert abs(hostparts-loop['host_union_ms'])<1e-5
    gpuwrap=gather['gpu_union_ms']+weight['gpu_union_ms'];gpupred=tau+gpuwrap
    # Conservatively compare tau plus observed non-compiled host budget. No
    # arbitrary fitted scale or common offset; gaps/overlap remain explicit.
    hostpred=tau+hostparts-compiled['host_union_ms']
    expert.append(dict(cache=cache,policy=policy,event=event,rank=rank,experts=len(calls),rows=sum(x['rows'] for x in calls),tau_ms=tau,compiled_gpu_ms=compiled['gpu_union_ms'],compiled_host_call_ms=compiled['host_union_ms'],loop_host_ms=loop['host_union_ms'],loop_gpu_span_ms=loop['gpu_span_ms'],loop_gpu_active_ms=loop['gpu_union_ms'],ready_wait_host_ms=ready,slot_gpu_wait_upper_bound_ms=sum(x['gpu_wait_upper_bound_ms'] for x in r['slot_wait_bounds']),ready_select_without_wait_ms=select,gather_host_ms=gather['host_union_ms'],gather_gpu_ms=gather['gpu_union_ms'],weight_host_ms=weight['host_union_ms'],weight_gpu_ms=weight['gpu_union_ms'],record_use_host_ms=record['host_union_ms'],loop_other_host_ms=other,pred_gpu_active_ms=gpupred,pred_loop_host_ms=hostpred,unexplained_host_budget_ms=loop['host_union_ms']-hostpred,unexplained_gpu_active_ms=loop['gpu_union_ms']-gpupred))
    timeline.append(dict(cache=cache,policy=policy,**r))
   for label,phase,packet,dst in [('forward','forward_collective_host_call',4120,forward),('return','return_collective_host_call',4096,back)]:
    ph=[r['phases'][phase] for r in rs];gs=np.array([x['gpu_start_ns'] for x in ph],np.int64);ge=np.array([x['gpu_end_ns'] for x in ph],np.int64);hs=np.array([x['host_start_ns'] for x in ph],np.int64);he=np.array([x['host_end_ns'] for x in ph],np.int64)
    co=model['transport'][str(packet)]['coefficients_ms'];base=float(co[0]+co[1]*max(S.sum(0)+S.sum(1)))
    service_ready=np.array(old['pred_expert_rank_ms']);service_wait=service_ready.max()-service_ready
    arrival=(gs.max()-gs)/1e6;hostarrival=(hs.max()-hs)/1e6;actual=np.array([x['gpu_union_ms'] for x in ph]);residual=actual-base-arrival
    readyphase='forward_pack_gpu' if label=='forward' else 'return_partial_build_gpu';readyts=np.array([r['phases'][readyphase]['gpu_end_ns'] for r in rs],np.int64)
    for rank in range(4):
     dst.append(dict(cache=cache,policy=policy,event=event,rank=rank,wire_hat_ms=base,nccl_residency_ms=float(actual[rank]),inflation_ms=float(actual[rank]-base),host_entry_raw_ns=int(hs[rank]),host_exit_raw_ns=int(he[rank]),gpu_start_raw_ns=int(gs[rank]),gpu_end_raw_ns=int(ge[rank]),host_arrival_spread_ms=float(np.ptp(hs)/1e6),gpu_start_spread_ms=float(np.ptp(gs)/1e6),ready_spread_ms=float(np.ptp(readyts)/1e6),ready_to_gpu_start_ms=float((gs[rank]-readyts[rank])/1e6),host_entry_to_gpu_start_ms=float((gs[rank]-hs[rank])/1e6),arrival_wait_hat_ms=float(arrival[rank]),service_wait_hat_ms=float(service_wait[rank]) if label=='return' else 0.,host_arrival_wait_hat_ms=float(hostarrival[rank]),residual_ms=float(residual[rank]),predicted_residency_ms=float(base+arrival[rank])))
  cells.append(dict(cache=cache,policy=policy,capture=str(cap)))
 def save(name,rows):
  (PACKET/(name+'.json')).write_text(json.dumps(dict(rows=rows),separators=(',',':'))+'\n')
  keys=[k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
  with (PACKET/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 save('B2_FORWARD_ATTRIBUTION',forward);save('B2_RETURN_ATTRIBUTION',back);save('B2_EXPERT_ATTRIBUTION',expert)
 # Deep per-expert intervals are retained locally, summarized portable timeline.
 raw=B2/'event_timeline.json';raw.write_text(json.dumps(timeline,separators=(',',':'))+'\n');sources[str(raw)]=sha(raw)
 slim=[{k:v for k,v in r.items() if k not in ('expert_calls','ready_wait_calls','slot_wait_bounds')} for r in timeline];save('B2_EVENT_TIMELINE',slim)
 summaries=[]
 for cell in cells:
  c,p=cell['cache'],cell['policy'];summary=dict(cache=c,policy=p)
  for name,data in [('forward',forward),('return',back)]:
   xs=[x for x in data if x['cache']==c and x['policy']==p];inflation=sum(max(0,x['inflation_ms']) for x in xs);err=sum(abs(x['residual_ms']) for x in xs)
   summary[name]=dict(mean_rank_residency_ms_per_step=sum(x['nccl_residency_ms'] for x in xs)/4/8,wire_ms_per_step=sum(x['wire_hat_ms'] for x in xs)/4/8,arrival_ms_per_step=sum(x['arrival_wait_hat_ms'] for x in xs)/4/8,service_wait_ms_per_step=sum(x['service_wait_hat_ms'] for x in xs)/4/8,signed_residual_ms_per_step=sum(x['residual_ms'] for x in xs)/4/8,absolute_residual_ms_per_step=err/4/8,explained_inflation_fraction=1-err/inflation,gate80=bool(err/inflation<=.2),arrival_residency_spearman=float(spearmanr([x['predicted_residency_ms'] for x in xs],[x['nccl_residency_ms'] for x in xs]).statistic),host_gpu_arrival_spread_spearman=float(spearmanr([x['host_arrival_spread_ms'] for x in xs],[x['gpu_start_spread_ms'] for x in xs]).statistic))
  xs=[x for x in expert if x['cache']==c and x['policy']==p];summary['expert']={k:sum(x[k] for x in xs)/4/8 for k in xs[0] if k.endswith('_ms')}
  ep=summary['expert'];ep['gpu_active_relative_error']=abs(ep['pred_gpu_active_ms']-ep['loop_gpu_active_ms'])/ep['loop_gpu_active_ms'];ep['host_loop_relative_error']=abs(ep['pred_loop_host_ms']-ep['loop_host_ms'])/ep['loop_host_ms'];ep['gate25']=bool(ep['gpu_active_relative_error']<=.25 and ep['host_loop_relative_error']<=.25)
  summaries.append(summary)
 invariance={};deltas={}
 for c in ('C30','C60'):
  br=next(s for s in summaries if s['cache']==c and s['policy']=='BR');fca=next(s for s in summaries if s['cache']==c and s['policy']=='FCA');deltas[c]={}
  for name in ('forward','return'):
   a,b=[s[name]['absolute_residual_ms_per_step'] for s in (br,fca)];gap=abs(a-b)/max((a+b)/2,1e-12);invariance[c+'/'+name]=dict(relative_difference=gap,within10=gap<=.1)
   deltas[c][name]={k:fca[name][k]-br[name][k] for k in br[name] if k.endswith('_step')}
  deltas[c]['expert']={k:fca['expert'][k]-br['expert'][k] for k in br['expert'] if k.endswith('_ms')}
 mechanism=all((s[name]['gate80'] or invariance[s['cache']+'/'+name]['within10']) for s in summaries for name in ('forward','return')) and all(s['expert']['gate25'] for s in summaries)
 # Measured arrival skew is not automatically placement-causal: source terms
 # mix CPU scheduling/controller/enqueue with GPU service. Require evidence,
 # not an algebraic subtraction, before calling the unexplained budget A.
 causal_support={}
 for cache in ('C30','C60'):
  d=deltas[cache]['return'];measured=d['arrival_ms_per_step'];pred=d['service_wait_ms_per_step']
  causal_support[cache]=dict(measured_BR_FCA_arrival_delta_ms_per_step=measured,predicted_current_expert_service_wait_delta_ms_per_step=pred,explained_delta_fraction=pred/measured if measured>0 else None,dominant_demonstrated=bool(measured>0 and pred/measured>=.5 and pred/measured<=1.25))
 placement=mechanism and all(x['dominant_demonstrated'] for x in causal_support.values())
 if placement:raise RuntimeError('B2 may support service-causal dominance; inspect evidence and run authorized revised-model validation before finalizing, rather than automatically reporting blocked.')
 result=dict(status='DIAGNOSTIC_COMPLETE_ORACLE_BLOCKED',mechanism_gate=mechanism,placement_causality='NOT_ESTABLISHED',current_service_causal_support=causal_support,revised_model_run=False,stage_c_authorized=False,summary=summaries,policy_residual_invariance=invariance,br_to_fca_deltas=deltas,clock_alignment=clocks,source_sha256=sources,classification=[dict(term='compiled expert service / owner row-and-expert allocation',category='A',evidence='exact expert ownership and n_e; measured compiled GPU service compared with frozen tau',limitation='Only this service term is predictable from the current Stage-A model.'),dict(term='slot wait host span',category='B_UNRESOLVED_A_CANDIDATE',evidence='measured waits refer to actual owner slots and exact cache parity',limitation='Host wait combines page-to-pinned staging submission and scheduling; GPU wait on completion can be asynchronous. Cannot assert all duration is admission-controllable.'),dict(term='host expert wrapper / selection / kernel enqueue / other',category='B',evidence='measured host phase budget and BR/FCA deltas',limitation='No independent intervention isolates Python scheduling/enqueue backpressure from deterministic owner-count effects.'),dict(term='wire/base collective service',category='A_SMALL_AND_C_BASE',evidence='frozen A1 intercept plus directed incident feature; no refit'),dict(term='arrival skew explaining NCCL residency',category='A_OR_B_DEPENDING_SOURCE',evidence='measured aligned GPU start and host entry spreads',limitation='Arrival agreement alone does not identify why the late rank is late.'),dict(term='pack / unpack / partial / combine',category='B_OR_C_UNPROVEN',evidence='separate actual GPU and host ranges retained',limitation='Do not subtract a common constant without cross-policy invariance evidence.')],limitations=['No timing exclusions or latency-selected tail events; prefix is fixed8steps.','B2 instrumentation changes host scheduling; no primary TPOT claim.','Host ready_wait measures submission blocking and wait insertion, not a direct isolated GPU DMA-stall duration.','Expert host budget and GPU-active accounting are distinct; adding them is forbidden.','Class-A dominance remains unproven; successful residency accounting is not sufficient for an oracle.'])
 write(PACKET/'B2_PLACEMENT_CAUSALITY.json',result)
 lines=['# Stage B2 runtime gap attribution','',f"Mechanism gate: **{mechanism}**. Placement causality: **not established**. Stage C remains blocked; no revised delta model or oracle executed.",'','Four Env1 R4/B128 C30/C60 × BR/FCA captures, first8decode steps. All capture runs must pass unchanged token/cache/copy/controller checks. No samples dropped. GPU2/3/6/7 remain owner-disabled.','', '## Collective residency accounting','', '| Cache | Policy | Phase | Residency ms/step | Base | Measured start-skew wait | Signed residual | Inflation explained |', '|---|---|---|---:|---:|---:|---:|---:|']
 for s in summaries:
  for phase in ('forward','return'):
   x=s[phase];lines.append(f"| {s['cache']} | {s['policy']} | {phase} | {x['mean_rank_residency_ms_per_step']:.3f} | {x['wire_ms_per_step']:.3f} | {x['arrival_ms_per_step']:.3f} | {x['signed_residual_ms_per_step']:.3f} | {x['explained_inflation_fraction']:.1%} |")
 lines+=['', 'Values are mean-rank ms per step, not additive TPOT. Explained fraction uses absolute per-event residual to prevent cancellation. Start-skew wait is max(rank GPU NCCL start) minus that rank start; it is not summed again with host skew/ready skew. Residual remains signed and all samples remain.','', '## Expert loop accounting','', '| Cache | Policy | Tau | Actual compiled GPU | Wrapper GPU | Host loop | Host ready wait | Host compiled call | Host other | GPU error | Host error |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for s in summaries:
  x=s['expert'];lines.append(f"| {s['cache']} | {s['policy']} | {x['tau_ms']:.3f} | {x['compiled_gpu_ms']:.3f} | {x['gather_gpu_ms']+x['weight_gpu_ms']:.3f} | {x['loop_host_ms']:.3f} | {x['ready_wait_host_ms']:.3f} | {x['compiled_host_call_ms']:.3f} | {x['loop_other_host_ms']:.3f} | {x['gpu_active_relative_error']:.1%} | {x['host_loop_relative_error']:.1%} |")
 lines+=['', 'Slot DMA completion is matched by expert key and submission order. A GPU wait upper bound uses prior same-stream work, host wait entry, DMA completion and the next gather kernel; it is not asserted to be an isolated stall measurement. Values remain in EXPERT_ATTRIBUTION.', '', 'Host and GPU figures are different clocks/interval domains and overlap. GPU estimate = tau + measured gather/weight GPU service. Host budget estimate = tau + measured non-compiled host budget; its error exposes kernel enqueue/backpressure beyond service time. Exact host phase partition is an accounting identity, not proof of causality. Ready-select includes ready-wait; the two are not double counted.','', '## Interpretation','', 'A near-one-for-one link between arrival skew and NCCL residency can explain where kernels wait without explaining why ranks arrive late. Expert ownership is exact and service time is measurable, but host enqueue, ready-slot submission, scheduling and staging remain mixed. These terms cannot be labeled placement-controllable solely because their BR/FCA averages differ. No global fitted scale or arbitrary common-offset subtraction is used.', '', 'Per-term BR→FCA deltas, residual policy-invariance checks, phase timelines and clock-calibration uncertainty are in the JSON/CSV artifacts. Any still-unidentified timing remains explicit, rather than assigned to wire service or expert service.', '', '## Stop', '', 'Stop for owner review per STAGE_B2_RUNTIME_GAP_ATTRIBUTION.md. This diagnostic does not authorize a new controller or oracle.']
 lines+=['']+['- '+s for s in result['limitations']]
 (PACKET/'B2_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(mechanism_gate=mechanism,summary=summaries,clock_p99_us=max(x['p99_anchor_residual_us'] for x in clocks)),indent=2))
if __name__=='__main__':report()
