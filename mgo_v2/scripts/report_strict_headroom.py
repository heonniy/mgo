"""Primary repeated TTFT and separate strict-phase diagnostic evidence."""
import statistics,json
from strict_headroom_common import *
def main():
 raw=json.loads((PACKET/'STRICT_HEADROOM_RESULTS.json').read_text());assert raw['status']=='PASS'
 rows=[]
 for final in raw['finals']:
  spec=final['row'];world=spec['world'];batch=spec['batch'];context=spec['context'];entry=dict(world=world,batch=batch,context=context,seed=spec,comparisons={})
  for policy,suffix in [('LA_CA_NEAR','LACA'),('LA','LA')]:
   label=f'S2_R{world}_B{batch}_L{context}_{suffix}';out=ROOT/'physical'/label
   rr=final[policy]['results'];assert len(rr)==1 and rr[0]['status']=='PASS';rr=rr[0]
   assert all(2<=len(v)<=3 for v in rr['samples'].values())
   diagnostics={}
   for p in ['BR',policy]:
    ranks=[json.loads((out/f'{label}_{p}_diagnostic_rank{r}.json').read_text()) for r in range(world)]
    assert all(x['status']=='PASS' and len(x['phases'])==48 for x in ranks)
    metrics=['h2d_local_ms','h2d_barrier_ms','forward_cuda_ms','expert_cuda_ms','compute_local_complete_ms','compute_barrier_ms','return_cuda_ms','controller_wall_ms','routing_admission_layout_ms']
    diagnostics[p]={m:statistics.mean(sum(layer[m] for layer in x['phases']) for x in ranks) for m in metrics}
    diagnostics[p]['rank_H2D_copies']=[x['validation']['observed']['copies'] for x in ranks]
    diagnostics[p]['rank_expert_rows']=[sum(x['validation']['expert_rows']) for x in ranks]
   assert diagnostics['BR']['rank_H2D_copies']==diagnostics[policy]['rank_H2D_copies']
   entry['comparisons'][policy]=dict(samples=rr['samples'],estimates=rr['estimates'],ranges={p:[min(v),max(v)] for p,v in rr['samples'].items()},gain=rr['gain'],unstable=rr['unstable'],decision='UNSTABLE' if rr['unstable'] else 'GO' if rr['gain']>=.05 else 'MARGINAL' if rr['gain']>=.02 else 'LOW_HEADROOM',diagnostics_mean_rank_ms=diagnostics)
  rows.append(entry)
 write(PACKET/'ANALYSIS.json',dict(status='PASS',rows=rows,scope='Deliberately BR-adversarial best-seed headroom; not average-case. S1 excluded from final estimate. Separate diagnostics are not additive global critical-path timing. LA and LA_CA_NEAR use the same selected triple, but separate processes and separately paired BR baselines.'))
 lines=['# Strict LA_CA_NEAR headroom results','','C30, BF16, cold prefill, no substitution/replication/prefetch. H2D and expert work finish with global barriers before the next phase. Deliberately BR-adversarial seed search; not an average-case benchmark. S1 selection samples are excluded from these fresh-process S2 estimates.','','| R | Batch | Context | Candidate | BR TTFT s | Candidate TTFT s | Gain | Decision |','|---:|---:|---:|---|---:|---:|---:|---|']
 for row in rows:
  for p,x in row['comparisons'].items():lines.append(f"| {row['world']} | {row['batch']} | {row['context']} | {p} | {x['estimates']['BR']:.4f} | {x['estimates'][p]:.4f} | {100*x['gain']:+.2f}% | {x['decision']} |")
 lines+=['','All valid repeats and their ranges, fixed selected seeds, copy/work accounting and separate phase diagnostics are preserved in ANALYSIS.json. No single-shot primary gains or latency-based sample deletion. Controller timings are separated from routing/admission/layout host time; return timing excludes the compute barrier. Phase times from diagnostic passes are not additive to primary TTFT.','']
 (PACKET/'ANALYSIS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
