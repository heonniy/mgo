"""Report physical model numerical differences without claiming free-run parity."""
import json
from pathlib import Path
from ca_stress_common import write
P=Path(__file__).resolve().parents[1];PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
def main():
 rows=[]
 for batch in [128,256]:
  path=ROOT/f'M12_PARTIAL_B{batch}';status=json.loads((path/'status.json').read_text());assert status['status']=='PASS'
  ranks=[json.loads((path/f'rank{r}.json').read_text()) for r in range(8)]
  for mode in ['bf16','fp32','fp64']:
   for policy in ['BR','LA']:
    reports=[next(x for x in r['reports'] if x['policy']==policy and x['precision']==mode) for r in ranks]
    assert all(x['copy_accounting']=='PASS' and x['logical_state']=='PASS' and x['payload_collectives']==768 for x in reports)
    total=sum(x['tokens'] for r in reports for x in r['steps']);equal=sum(x['argmax_equal'] for r in reports for x in r['steps'])
    rows.append(dict(batch=batch,policy=policy,precision=mode,legacy_argmax_agreement=equal/total,differing_argmax=total-equal,total_argmax=total,max_relative_L2=max(r['max_relative_L2'] for r in reports),max_abs_logit_difference=max(r['max_abs'] for r in reports),forward_bytes=sum(r['forward_bytes'] for r in reports),return_bytes=sum(r['return_bytes'] for r in reports),BR_LA_argmax_agreement=(sum(r['BR_LA_argmax_agreement'] for r in reports)/8 if policy=='LA' else None)))
 write(PACKET/'M12_PARTIAL_MODEL_SUMMARY.json',dict(status='PASS',rows=rows,horizon=8,scope='prefill plus eight fixed decode steps; numerical/argmax agreement only, not free-generation quality or speed',all_copy_state_and_collective_gates='PASS',precision_selection='FP32 common main-stack candidate: synthetic physical transport exactly matches high-precision contribution sum, at half the FP64 return bytes. BF16 remains an eligible separate physical candidate under owner amendment. No LA timing used.'))
 lines=['# Coalesced return numerical validation','','One prefill plus eight frozen decode steps, all eight GPUs. Numbers compare final logits and argmax against the legacy BF16 expert-order reference. Teacher inputs and router selections/weights stay frozen. These are not free-generation quality results.','','| Batch | Policy | Accumulation | Argmax agreement | Different / total | Max relative L2 | Max absolute logits delta |','|---|---|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f'| {r["batch"]} | {r["policy"]} | {r["precision"]} | {r["legacy_argmax_agreement"]:.4%} | {r["differing_argmax"]} / {r["total_argmax"]} | {r["max_relative_L2"]:.6f} | {r["max_abs_logit_difference"]:.6f} |')
 lines+=['','Every candidate preserves the CPU cache/role/copy plan and uses exactly two payload A2As per decode layer. The return contains one partial vector per received token/rank packet.','', 'Use FP32 accumulation as the common three-arm stack. In the independent transport tests it matched the canonical high-precision contribution sum, while costing half the FP64 return bytes. BF16 remains a separately reported candidate; it is not silently excluded because of rounding differences. No placement-policy performance measurements were used for this precision choice.']
 (PACKET/'M12_PARTIAL_MODEL_SUMMARY.md').write_text('\n'.join(lines)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
