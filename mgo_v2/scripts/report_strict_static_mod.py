"""Add sequential static-owner baseline to all eight strict decode32 cells."""
from strict_headroom_common import *
def main():
 rows=[]
 for world in (4,8):
  for batch in (16,64):
   for context in (256,512):
    d=ROOT/'physical'/f'STATIC32_R{world}_B{batch}_L{context}';s=json.loads((d/'status.json').read_text());assert s['status']=='PASS';x=s['result'];assert len(x['samples']['STATIC_MOD']) in (2,3)
    old=json.loads(Path(x['reference_result']).read_text());proof=json.loads((Path(x['spec']['inputs'])/'receipt.json').read_text())['proofs']['STATIC_MOD']
    original_dir=Path(x['reference_result']).parent;baseline_copies={}
    for p in ('BR','LA_CA_NEAR','LA'):
     receipts=[json.loads((original_dir/f'{p}_r1_measure_rank{r}.json').read_text()) for r in range(world)]
     baseline_copies[p]={phase:[z['validation'][phase+'_H2D_copies'] for z in receipts] for phase in ('prefill','decode')}
    static_receipts=[json.loads((d/f'STATIC_MOD_r1_measure_rank{r}.json').read_text()) for r in range(world)]
    assert all(z['validation']['tokens_match_reference_BR'] for z in static_receipts)
    baseline_copies['STATIC_MOD']={phase:[z['validation'][phase+'_H2D_copies'] for z in static_receipts] for phase in ('prefill','decode')}
    rows.append(dict(world=world,batch=batch,context=context,static=x,original=old,all_estimates=dict(old['estimates'],STATIC_MOD=x['estimates']['STATIC_MOD']),copies_by_rank=baseline_copies,static_vs_BR_work_class='STRICT_PLACEMENT' if all(sum(baseline_copies['STATIC_MOD'][phase])==sum(baseline_copies['BR'][phase]) for phase in ('prefill','decode')) else 'PLACEMENT_STATE_EFFECT',static_owner='expert_id % world',copy_quota_balanced=False))
 write(PACKET/'STATIC_MOD_RESULTS.json',dict(status='PASS',rows=rows,scope='Static added sequentially on identical selected seeds and frozen routes/tokens, not an interleaved timing pair. Earlier BR/LA_CA_NEAR/LA results are historical references; possible session drift. Static misses have no balanced-count quota. Same capacity, eviction, strict barriers, precision and decode32. No diagnostic passes.'))
 lines=['# Static owner baseline: eight strict decode32 conditions','','STATIC_MOD fetches each missing expert to logical rank `expert_id % world`. R4 maps logical ranks0/1/2/3 to physical GPUs0/1/4/5; R8 maps directly0..7. Same selected inputs and frozen traces, cache capacity and eviction; no replica/substitution/prefetch.','','Static is a **sequential addition**. Original BR/LA_CA_NEAR/LA were not rerun; comparisons can include session drift. Each static estimate has two repeats (mean) or one bounded third (median), with all samples retained. No extra diagnostic passes.','','| R | B | Input | Policy | TTFT s | TPOT s/token | E2E s |','|---:|---:|---:|---|---:|---:|---:|']
 for x in rows:
  for p in ('BR','LA_CA_NEAR','LA','STATIC_MOD'):
   e=x['all_estimates'][p];lines.append(f"| {x['world']} | {x['batch']} | {x['context']} | {p} | {e['TTFT']:.4f} | {e['TPOT']:.4f} | {e['E2E']:.4f} |")
 lines+=['','All repeats, instability flags, earlier reference results, and per-rank prefill/decode copy counts are retained in STATIC_MOD_RESULTS.json. Static placement does not balance miss counts; differences in physical H2D work are classified as placement-state effects. No stable performance claim is supported when either the static metric or reference metric is flagged unstable.','']
 (PACKET/'STATIC_MOD_RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
