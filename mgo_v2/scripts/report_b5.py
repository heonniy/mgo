"""B5 clean timing and separate, non-additive barrier isolation diagnostics."""
import json,statistics,subprocess
from pathlib import Path
from prepare_critical_microbench import ROOT,PACKET,P
from refactor_fingerprint import sha
BASE=ROOT/'b5/C30'
NSYS='/home/hwlee/mgo-tools/nsight-2025.3/extracted/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys'
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def main():
 clean=BASE/'B5_C30_CLEAN_B128_H64';state=json.loads((clean/'status.json').read_text());assert state['status']=='PASS'
 identity=state['common_stack']
 assert all(sha(P/n)==v for n,v in identity['code'].items()),'measured code changed'
 assert all(sha(BASE/'inputs_B128_H64'/n)==v for n,v in identity['inputs'].items()),'input changed'
 result=json.loads((clean/'result.json').read_text());correct=[]
 for policy in ('BR','FCA'):
  for rank in range(4):
   r=json.loads((clean/f'{policy}_H1b_correctness_rank{rank}.json').read_text());assert r['status']=='PASS' and r['post_expert_barriers']==3072
   correct.append(dict(policy=policy,rank=rank,**r))
 write(PACKET/'B5_CORRECTNESS.json',dict(status='PASS',rows=correct))
 write(PACKET/'B5_TIMING_REPEATS.json',result)
 diagnostics=[]
 for policy in ('BR','FCA'):
  cap=BASE/f'B5_C30_H1b_V3_OPT_PF_OVERLAP_{policy}_B128_H8'
  assert json.loads((cap/'status.json').read_text())['status']=='PASS'
  out=cap/'interval_analysis'
  if not (out/'summary.json').exists():
   assert not out.exists(),'preserve partial export'
   with (cap/'export_driver.log').open('w') as f:
    subprocess.run(['/home/hwlee/sub-moe/phase01/.venv/bin/python',str(P/'scripts/export_refactor_profiles.py'),str(cap),'--nsys',NSYS],stdout=f,stderr=subprocess.STDOUT,check=True)
  ranks=[]
  for rank in range(4):
   check=json.loads((cap/f'b5_capture_validation_rank{rank}.json').read_text());assert check['status']=='PASS' and check['post_expert_barriers']==384
   full=json.loads((cap/f'b5_full64_validation_rank{rank}.json').read_text());assert full['status']=='PASS'
   x=json.loads((out/f'rank{rank}_intervals.json').read_text())
   times={name:x['cpu_nvtx'][name]['total_ms']/8 for name in ('moe.post_expert_local_complete','moe.post_expert_global_barrier','moe.return_a2a')}
   calibration=json.loads((cap/f'b5_barrier_calibration_rank{rank}.json').read_text())
   ranks.append(dict(rank=rank,host_ms_per_step=times,return_nccl_residency_ms_per_step=x['kernel_union_ms']['moe.return_a2a.nccl']/8,barrier_nccl_residency_ms_per_step=x['kernel_union_ms'].get('moe.post_expert_global_barrier.nccl',0)/8,idle_barrier_reference_ms=statistics.median(t['median_ms'] for t in calibration['trials']),unassigned_decode_kernel_count=x['unassigned_decode_kernel_count']))
  mean={k:statistics.mean(r['host_ms_per_step'][k] for r in ranks) for k in ranks[0]['host_ms_per_step']}
  tpot=result['gates'][policy+'_H1b']['estimate']['TPOT']
  diagnostics.append(dict(policy=policy,ranks=ranks,mean_rank_host_ms_per_step=mean,mean_rank_host_fraction_of_clean_TPOT={k:v/(1000*tpot) for k,v in mean.items()},normalization_valid=not result['gates'][policy+'_H1b']['unstable'],mean_rank_idle_barrier_reference_ms_per_step=48*statistics.mean(r['idle_barrier_reference_ms'] for r in ranks)))
 write(PACKET/'B5_DIAGNOSTIC_RESULTS.json',dict(status='PASS',rows=diagnostics,primary_timing=False,interpretation='First8 instrumented prefix, mean rank-local host durations. TPOT denominator is separate full64 clean measurement. Fractions are descriptive and not an additive global critical-path decomposition. Barrier includes collective overhead and scheduling; idle reference cannot be subtracted as exact imbalance.'))
 br=result['gates']['BR_H1b']['estimate'];fca=result['gates']['FCA_H1b']['estimate']
 gain=1-fca['TPOT']/br['TPOT']
 lines=['# B5 post-expert global barrier results','',f"Status: {'UNSTABLE_TIMING' if result['unstable'] else 'COMPLETE'}. C30, local B128, R4 GPUs 0/1/4/5, frozen decode64, BF16 H1b V3/P2/T2.",'','Canonical token, controller, H2D copy/byte and payload parity passed. Every full64 rank performed exactly 3072 post-expert barriers. No copy-parity relaxation or automatic retry.','','| Policy | TPOT (s) | E2E (s) | Repeats | TPOT range (s) |','|---|---:|---:|---:|---|']
 for policy in ('BR','FCA'):
  g=result['gates'][policy+'_H1b'];lines.append(f"| {policy} | {g['estimate']['TPOT']:.6f} | {g['estimate']['E2E_wall']:.6f} | {len(result['samples'][policy+'_H1b'])} | {g['range']['TPOT']} |")
 lines+=['',f"FCA TPOT gain versus BR: {100*gain:.3f}% (negative means slower)." if not result['unstable'] else 'Timing is unstable: do not interpret the descriptive estimates as an accepted gain.','','## Separate first8 diagnostic','','| Policy | Local completion (ms/step) | Global barrier (ms/step) | Return host range (ms/step) | Idle barrier reference ×48 (ms/step) |','|---|---:|---:|---:|---:|']
 for d in diagnostics:
  v=d['mean_rank_host_ms_per_step'];lines.append(f"| {d['policy']} | {v['moe.post_expert_local_complete']:.3f} | {v['moe.post_expert_global_barrier']:.3f} | {v['moe.return_a2a']:.3f} | {d['mean_rank_idle_barrier_reference_ms_per_step']:.3f} |")
 lines+=['','These are mean rank-local instrumented durations, not a sum of global critical-path penalties. Return host duration and actual return NCCL residency are different quantities; both are preserved in B5_DIAGNOSTIC_RESULTS.json. The idle barrier reference includes scheduling and is not an exact service-cost subtraction. First8 diagnostic/full64 TPOT normalization is descriptive.','','The barrier is an isolation tool, not an accepted production optimization. Historical no-barrier timings are not a contemporaneous causal control. No C60, R8, H2 or Stage C extension was run.','']
 (PACKET/'B5_RESULTS.md').write_text('\n'.join(lines))
 write(PACKET/'B5_AUDIT.json',dict(status='PASS',measured_code_hashes_unchanged=True,frozen_input_hashes_unchanged=True,correctness_rows=len(correct),all_samples_retained=True,primary_timing_stable=not result['unstable'],physical_gpus=[0,1,4,5],clean_source_sha=state['source_sha'],result= str(clean/'result.json')))
if __name__=='__main__':main()
