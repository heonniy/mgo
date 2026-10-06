"""Complete bounded B4 results, explicit mechanism gates and physical repeats."""
import json,csv,statistics
from pathlib import Path
from prepare_critical_microbench import ROOT,PACKET,write,sha
from physical_repeat_rule import decide,final_unstable

def main():
 cap=ROOT/'b4/C30/B4_C30_CLEAN_RETRY1_B128_H64';result=json.loads((cap/'result.json').read_text());assert result['status']=='PASS'
 mechanism=json.loads((PACKET/'B4_KNOB_RUNTIME_GAP.json').read_text());correctness=json.loads((PACKET/'B4_CORRECTNESS.json').read_text())
 timing=[];summary={}
 for key,observations in result['samples'].items():
  policy,mode=key.split('_');summary[key]={metric:statistics.median(r[metric] for r in observations) for metric in ('TPOT','E2E_wall')}
  assert len(observations)==decide(observations[:2])['target_repeats']
  for i,row in enumerate(observations,1):
   raw=[json.loads((cap/f'{key}_r{i}_measure_rank{rank}.json').read_text()) for rank in range(4)]
   assert all(r['status']=='PASS' and r['no_compile_in_measure'] for r in raw)
   for metric in ('TPOT','E2E_wall'):assert row[metric]==max(r[metric] for r in raw)
   timing.append(dict(policy=policy,executor=mode,repeat=i,**row))
 gates={};gains={}
 for policy in ('BR','FCA'):
  m=mechanism['gates'][policy]
  gates[policy]=dict(counter_parity=m['parity'],host_call_reduction_80=m['host_call_reduction']>=.8,host_loop_reduction_25=m['host_loop_reduction']>=.25,service_spearman_080=m['service_stage_spearman'] is not None and m['service_stage_spearman']>=.8,timing_stable=not any(final_unstable(result['samples'][f'{policy}_{mode}']) for mode in ('H1b','H2')))
  gains[policy]={metric:1-summary[policy+'_H2'][metric]/summary[policy+'_H1b'][metric] for metric in ('TPOT','E2E_wall')}
 common=dict(correctness=correctness['correctness_pass'],no_signature_graph_required=all(r['executor']['graph_entries']==0 for r in correctness['ranks']),br_regression_at_most_1pct=gains['BR']['TPOT']>=-.01,delta_error_at_most_25pct=mechanism['delta']['relative_error']<=.25)
 passed=all(common.values()) and all(all(g.values()) for g in gates.values())
 interpretation='A' if passed else ('UNSTABLE' if any(not g['timing_stable'] for g in gates.values()) else 'C' if all(g['TPOT']<=.01 for g in gains.values()) else 'B')
 decision=dict(case=interpretation,status='C30_H2_GATE_PASS' if passed else 'C30_H2_GATE_FAIL',gates=gates,common_gates=common,estimates=summary,gains=gains,mechanism=mechanism,c60_authorized=passed,oracle_authorized=False)
 write(PACKET/'B4_REPAIR_DECISION.json',decision)
 write(PACKET/'B4_TIMING_REPEATS.json',dict(status='PASS',samples=timing,summaries=result['gates'],source_hashes={str(p):sha(p) for p in sorted(cap.glob('*_measure_rank*.json'))}))
 with (PACKET/'B4_TIMING_REPEATS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(timing[0]),lineterminator="\n");w.writeheader();w.writerows(timing)
 lines=['# B4 C30 dynamic grouped executor results','',f"Decision: **{decision['status']}**, interpretation **{interpretation}**. No Stage C oracle authorized.",'','R4 GPUs 0,1,4,5; C30/local B128; BF16; frozen decode64; V3 P2/T2; Env1.','', '|Policy/runtime|TPOT seconds|E2E seconds|','|---|---:|---:|']
 for key,r in summary.items():lines.append(f"|{key}|{r['TPOT']:.6f}|{r['E2E_wall']:.6f}|")
 lines+=['','All valid samples retained; initial two repeats use the frozen <=2% / <=5% rule.','Full ranges and stability decisions are in B4_TIMING_REPEATS.json.','', '## Mechanism gate','']
 for policy,m in mechanism['gates'].items():lines.append(f"- {policy}: host dispatch count reduction {100*m['host_call_reduction']:.2f}%; host-loop reduction {100*m['host_loop_reduction']:.2f}%; grouped service/expert-stage Spearman {m['service_stage_spearman']}.")
 lines+=['',f"BR->FCA expert-stage delta relative prediction error: {100*mechanism['delta']['relative_error']:.2f}%. No readiness exception automatically applied.",'','Correctness, workspace, micro-calibration, per-event diagnostics and provenance are in the corresponding B4 artifacts. H2 never reads future signatures. H1b uses its previously validated graph signatures as a reference. Clean paired runs retain the same H1b graph buffers in both modes; graph-free H2 memory is recorded separately in the correctness process.','', 'C60 may run only if the C30 gate passes. A failed mechanism criterion is not replaced by a TPOT-only claim.']
 (PACKET/'B4_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status=decision['status'],gains=gains,c60_authorized=passed)))
if __name__=='__main__':main()
