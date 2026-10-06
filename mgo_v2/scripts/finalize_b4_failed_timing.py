"""Close B4 after the bounded canonical-workload retry fails; no further runs."""
import csv,json,statistics
from pathlib import Path
from prepare_critical_microbench import ROOT,PACKET,P,write,sha


def main():
 base=ROOT/'b4/C30';attempt=base/'B4_C30_CLEAN_RETRY1_B128_H64'
 status=json.loads((attempt/'status.json').read_text());assert status['status']=='FAIL'
 original=json.loads((base/'B4_C30_CLEAN_B128_H64/status.json').read_text());assert original['status']=='FAIL'
 fp=status['common_stack']
 for name,digest in fp['code'].items():assert sha(P/name)==digest,('runtime source changed',name)
 for name,digest in fp['inputs'].items():assert sha(base/'inputs_B128_H64'/name)==digest
 warm=[json.loads(p.read_text()) for p in sorted(attempt.glob('*_correctness_rank*.json'))]
 assert len(warm)==16 and all(r['status']=='PASS' and r['counter_parity'] and r['token_parity'] and r['scheduler_metrics']['canceled']==0 for r in warm)
 validations=[json.loads((attempt/f'BR_H2_r1_validation_rank{r}.json').read_text()) for r in range(4)]
 assert all(r['no_compile'] and r['token_parity'] for r in validations)
 failure=[]
 for rank,row in enumerate(validations):
  reference=json.loads((attempt/f'BR_H2_correctness_rank{rank}.json').read_text())['reference']
  assert row['controller_counters']==reference['counters']
  actual=row['scheduler_metrics'];delta=actual['copies']-reference['copies']
  failure.append(dict(rank=rank,physical_gpu=[0,1,4,5][rank],token_parity=row['token_parity'],no_compile=row['no_compile'],expected_copies=reference['copies'],actual_copies=actual['copies'],copy_delta=delta,expected_bytes=reference['bytes'],actual_bytes=actual['bytes'],byte_delta=actual['bytes']-reference['bytes'],canceled=actual['canceled'],controller_equal=True))
 assert [r['copy_delta'] for r in failure]==[0,0,0,-1]
 assert failure[3]['byte_delta']==-9437184 and failure[3]['canceled']==1
 write(PACKET/'B4_FAST_PATH_COUNTER_FAILURE.json',dict(status='FAIL',sample='BR_H2_r1',ranks=failure,mechanism='An abandoned speculative prefetch can be canceled while still queued, before H2D submission. This preserves logical controller/token results but changes physical copy bytes. The first attempt also observed this in H1b, so it is not specific to grouped GEMM arithmetic.',primary_eligible=False,additional_retry_authorized=False))
 samples=[];raw=[]
 for key in ('BR_H1b_r1','FCA_H1b_r1'):
  rs=[json.loads((attempt/f'{key}_measure_rank{rank}.json').read_text()) for rank in range(4)]
  assert all(r['status']=='PASS' and r['no_compile_in_measure'] for r in rs)
  policy=key.split('_')[0]
  for rank,r in enumerate(rs):assert r['counters']==json.loads((attempt/f'{policy}_H1b_correctness_rank{rank}.json').read_text())['reference']
  samples.append(dict(policy=policy,executor='H1b',repeat=1,status='VALID_SINGLE_SAMPLE_ONLY',TPOT=max(r['TPOT'] for r in rs),E2E_wall=max(r['E2E_wall'] for r in rs),primary_eligible=False,reason='No second repeat and no valid paired H2 timing'))
  raw.extend(rs)
 partial=[str(p) for p in sorted(attempt.glob('BR_H2_r1_measure_rank*.json'))]
 assert len(partial)==3
 timing=dict(status='INCOMPLETE_COUNTER_PARITY_FAILURE',samples=samples,invalid_sample='BR_H2_r1',partial_rank_timing_paths=partial,primary_comparison_available=False,primary_gain=None,valid_samples_discarded=0,rule='Retain all receipts; no gain from single samples, incomplete ranks, or unequal H2D workload.')
 write(PACKET/'B4_TIMING_REPEATS.json',timing)
 with (PACKET/'B4_TIMING_REPEATS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(samples[0]),lineterminator='\n');w.writeheader();w.writerows(samples)
 corr=json.loads((PACKET/'B4_CORRECTNESS.json').read_text())
 assert len(corr['ranks'])==8 and all(r['status']=='PASS' for r in corr['ranks'])
 assert all(r['executor']['max_abs']==r['executor']['max_relative']==0 and r['executor']['graph_entries']==0 for r in corr['ranks'])
 assert sum(r['executor']['expert_comparisons'] for r in corr['ranks'] if r['policy']=='FCA')==648450
 corr.update(status='ARITHMETIC_PASS_FAST_PATH_COUNTER_REPRODUCIBILITY_FAIL',checked_full64_pass=True,correctness_pass=False,checked_pass_scope='Full64 expert-by-expert numerical checking slows/synchronizes the execution path; every checked output and token passed. This does not establish fast-path copy-count reproducibility.',fast_path_counter_failure=failure)
 write(PACKET/'B4_CORRECTNESS.json',corr)
 mechanism=json.loads((PACKET/'B4_KNOB_RUNTIME_GAP.json').read_text())
 decision=dict(status='H2_NOT_ACCEPTED',stop_reason='Strict physical H2D-copy parity failed after one bounded retry; mechanism gates also failed.',arithmetic_full64_pass=True,fast_path_counter_parity=False,diagnostic_gates=mechanism['gates'],delta_error=mechanism['delta'],primary_timing_complete=False,primary_gain=None,case='COUNTER_PARITY_STOP_BEFORE_TIMING_DECISION',c60_authorized=False,oracle_authorized=False,additional_gpu_runs_authorized=False)
 write(PACKET/'B4_REPAIR_DECISION.json',decision)
 paths=list(attempt.glob('*_measure_rank*.json'))+list(attempt.glob('*_validation_rank*.json'))+list(attempt.glob('*_correctness_rank*.json'))
 paths += [attempt/'status.json',base/'B4_C30_CLEAN_B128_H64/status.json',PACKET/'B4_GROUPED_SERVICE_CALIBRATION.json',PACKET/'CRITICAL_PATH_MODEL.json',PACKET/'STAGE_B4_GROUPED_GEMM_EXECUTOR.md']
 peak=max(r['peak_gpu_bytes'] for r in raw)
 audit=dict(status='PASS_FOR_BOUNDED_FAILURE_REPORT',physical_gpus=[0,1,4,5],valid_complete_single_samples=2,invalid_global_samples=1,primary_comparisons=0,warm_rank_receipts=16,checked_full64_rank_receipts=8,all_attempts_and_samples_retained=True,no_latency_based_exclusion=True,runtime_and_input_hashes_unchanged=True,standalone_workspace_bytes_per_gpu=69206016,minimum_observed_timing_host_available_bytes=min(b['sample']['host_available_bytes'] for b in status['boundaries']),maximum_gpu_allocated_bytes_in_complete_sample_receipts=peak,source_hashes={str(p):sha(p) for p in paths})
 write(PACKET/'B4_AUDIT.json',audit)
 wave_rows=json.loads((PACKET/'B4_WAVE_STATS.json').read_text())['rows'];distributions={}
 for policy in ('BR','FCA'):
  ws=[w for w in wave_rows if w['policy']==policy]
  distributions[policy]=dict(singleton_fraction=sum(w['experts']==1 for w in ws)/len(ws),mean_experts_per_wave=statistics.mean(w['experts'] for w in ws))
 lines=['# B4 dynamic grouped-GEMM executor validation','','**H2_NOT_ACCEPTED. No valid primary TPOT/E2E comparison or gain is available.**','','R4 physical GPUs 0,1,4,5; C30/local B128; BF16; frozen decode64; Env1; V3 P2/T2. No C60 or Stage C oracle was run.','', '## Completed implementation and correctness','','- Dynamic Triton ready-wave GEMM reads live cache weights, with no future signatures or graph cache. H0/H1b remain available and H2 remains opt-in.','- Standalone persistent workspace: 66 MiB/GPU plus small current-wave metadata.','- BR/FCA full64 checked passes: 648,450 expert comparisons, maximum absolute/relative difference 0, identical tokens and counters in those passes.','- Two CPU scheduler tests pass. Fifty-three isolated service calibration vectors and all four separate first8 captures completed.','', '## Why timing stopped','','The first clean preparation had one canceled 9-MiB BR/H1b prefetch on rank1. Its token hash still matched. Failed H2 rank1 values were not saved before its assertion; their exact outcome cannot be reconstructed. This attempt produced no primary timing samples.','','One bounded retry added canonical B3 workload checks and receipt-before-assert reporting. All 16 warm rank checks passed with zero canceled copies. BR/H1b and FCA/H1b each then produced one valid sample. During BR/H2 r1, rank3 (physical GPU5) canceled one prefetch: **55,022 copies instead of 55,023**, or **9 MiB fewer H2D bytes**. All four H2 token hashes and no-compilation checks passed, and controller counters matched. The strict physical-workload comparison failed.','','This is timing-dependent cancellation in the existing asynchronous prefetch scheduler, observed in both H1b and H2. It is not evidence of a grouped-GEMM arithmetic error. No forced copy, counter tolerance, or latency-based exclusion was introduced. No further retry was run.','','The two completed H1b samples and three partial H2 rank receipts are preserved in B4_TIMING_REPEATS.json and local raw artifacts. They do not meet the two-repeat paired requirement, so no performance gain is reported.','', '## Separate mechanism evidence','','|Policy|Host dispatch reduction|Host-loop reduction|Placement-row/service Spearman|Singleton waves|','|---|---:|---:|---:|---:|']
 for policy in ('BR','FCA'):
  g=mechanism['gates'][policy];d=distributions[policy];lines.append(f"|{policy}|{100*g['host_call_reduction']:.2f}%|{100*g['host_loop_reduction']:.2f}%|{g['service_stage_spearman']:.3f}|{100*d['singleton_fraction']:.2f}%|")
 lines += ['', 'Targets were >=80% host dispatch reduction, >=25% host-loop reduction, and Spearman >=0.80. These mechanism gates failed independently of the timing interruption. Host-loop values are single separate instrumented captures, not primary TPOT; do not interpret their differences as stable physical speedups.','','Observed-wave service sums correlate strongly with measured duration, but those wave boundaries are physical readiness outcomes. They are retained as post-hoc diagnostics and cannot pass the placement-visible prediction gate. H2D readiness fragments the immediate ready waves into mostly singleton work.','','## Resource and provenance audit','',f"Receipt audit passed for this bounded failure report. Minimum host available at timing boundaries: {audit['minimum_observed_timing_host_available_bytes']/2**30:.1f} GiB. Maximum GPU allocation recorded by complete samples: {peak/2**30:.2f} GiB. No OOM was observed. Only GPUs 0,1,4,5 were used; 2,3,6,7 remain reserved for other users.",'','All raw attempts, checked outputs, single timing samples, invalid-rank receipts, calibration vectors and diagnostic summaries are retained. No large Nsight file is committed. See B4_AUDIT.json, B4_FAST_PATH_COUNTER_FAILURE.json and B4_REPAIR_DECISION.json. Stop for owner review; no further GPU study is queued.']
 (PACKET/'B4_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status=decision['status'],valid_primary_comparisons=0,stop_reason=decision['stop_reason'])))
if __name__=='__main__':main()
