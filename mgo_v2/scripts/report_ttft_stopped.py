"""Audit and summarize completed S1 pairs after the owner's bounded stop."""
from ttft_common import *
import statistics

def main():
 state=json.loads((ROOT/'physical_status.json').read_text())
 assert state['error']=="RuntimeError('owner STOP')" and (ROOT/'STOP').exists()
 assert not list((ROOT/'physical').glob('TTFT_S2_*'))
 pairs=[];coverage=[];incomplete=[]
 for cache,batch in [(30,16),(60,16),(30,128),(60,128)]:
  d=ROOT/'physical'/f'TTFT_S1_C{cache}_B{batch}'
  specs=json.loads((d/'specs.json').read_text());done=[]
  for spec in specs:
   key=spec['key'];f=d/(key+'_result.json')
   if not f.exists():
    incomplete.append(dict(key=key,available_measure_receipts=[p.name for p in d.glob(key+'_*_measure_rank*.json')],included=False));continue
   x=json.loads(f.read_text());assert x['status']=='PASS' and x['selection_only'];candidate=spec['policy'];policies={}
   for policy in ('BR',candidate):
    ranks=[json.loads((d/f'{key}_{policy}_r1_measure_rank{k}.json').read_text()) for k in range(4)]
    for k,rank in enumerate(ranks):
     v=rank['validation'];assert rank['status']==v['status']=='PASS' and v['observed']==v['expected']
     checks=json.loads((d/f'{key}_{policy}_r1_validation_rank{k}.json').read_text());assert checks['no_compile'] and checks['tokens_match_warm']
     gate=json.loads((d/f'{key}_pair_gate_rank{k}.json').read_text());assert gate['status']=='PASS' and all(gate['tokens_match'])
    rows=[r['validation']['expert_rows'] for r in ranks]
    policies[policy]=dict(TTFT_s=max(r['TTFT'] for r in ranks),wire_bytes=sum(r['validation']['wire_bytes'] for r in ranks),H2D_bytes_by_rank=[r['validation']['observed']['bytes'] for r in ranks],copies_by_rank=[r['validation']['observed']['copies'] for r in ranks],expert_rows_by_rank=[sum(a) for a in rows],sum_layer_max_rank_rows=sum(max(a[l] for a in rows) for l in range(48)))
    assert abs(policies[policy]['TTFT_s']-x['estimates'][policy])<1e-9
   br=policies['BR'];p=policies[candidate]
   assert br['copies_by_rank']==p['copies_by_rank'] and br['H2D_bytes_by_rank']==p['H2D_bytes_by_rank']
   assert sum(br['expert_rows_by_rank'])==sum(p['expert_rows_by_rank'])==4*batch*512*8*48
   pairs.append(dict(cache=cache,batch=batch,candidate=candidate,key=key,workload_seed=spec['workload_seed'],placement_seed=spec['placement_seed'],gain_pct=x['gain']*100,wire_change_pct=100*(p['wire_bytes']/br['wire_bytes']-1),critical_rows_change_pct=100*(p['sum_layer_max_rank_rows']/br['sum_layer_max_rank_rows']-1),classification='STRICT_PLACEMENT',policies=policies,source=str(f),source_sha256=sha(f)))
   done.append(key)
  coverage.append(dict(cache=cache,batch=batch,completed=len(done),planned=len(specs),status='COMPLETE_S1' if len(done)==len(specs) else 'OWNER_STOPPED_PARTIAL'))
 summaries=[]
 for c in coverage:
  for policy in ('CA','OLD_CA','LA'):
   a=[r for r in pairs if (r['cache'],r['batch'],r['candidate'])==(c['cache'],c['batch'],policy)]
   best=max(a,key=lambda x:x['gain_pct'])
   summaries.append(dict(cache=c['cache'],batch=c['batch'],candidate=policy,n=len(a),best=best,gain_range_pct=[min(x['gain_pct'] for x in a),max(x['gain_pct'] for x in a)],median_screen_gain_pct=statistics.median(x['gain_pct'] for x in a)))
 result=dict(status='OWNER_STOPPED_ANALYZED',dataset='ShareGPT_LONG512',context=512,gpus=GPUS,coverage=coverage,summaries=summaries,pairs=pairs,incomplete_pairs=incomplete,primary_validated_gain=None,S2_run=False,phase_timing_attribution='NOT_MEASURED',stop_reason='Owner requested stop last condition, analyze existing samples and commit. No continuation or S2.',limitations=['S1 single-shot selection samples only; no repeated primary timing or stability claim.','Policy-specific best seeds; BR is paired within each comparison. Not average-case performance.','C60/B128 incomplete; unmatched/interrupted pairs excluded without latency-based filtering.','Row imbalance is measured work, not elapsed compute time. Controller, GEMM, synchronization and communication time shares not identified.','Cold expert cache: same H2D work across policies; cache capacity is not a repeated-decode hit-rate experiment.'])
 write(PACKET/'TTFT_STOPPED_RESULTS.json',result)
 lines=['# Owner-stopped ShareGPT_LONG512 TTFT analysis','','Status: OWNER_STOPPED_ANALYZED. The owner stopped the final C60/B128 condition and requested analysis of available results. S2 and phase diagnostics were not run. The generic runner labels the stop FAIL with `owner STOP`; this is an intentional cancellation, not a detected correctness/OOM failure. Raw receipts remain unchanged.','','## Scope and coverage','','R4 physical GPUs 0/1/4/5; BF16; exact512 valid tokens from 2048 frozen long ShareGPT conversation prefixes; empty expert cache; substitution/replication off. CA and OLD_CA are separate candidates, with paired BR and LA.','','| Cache | Local batch | Complete pairs / planned | Status |','|---|---:|---:|---|']
 for c in coverage:lines.append(f"| C{c['cache']} | {c['batch']} | {c['completed']}/{c['planned']} | {c['status']} |")
 lines+=['','## Best observed selection candidates','','These are single-shot S1 best-seed observations, **not validated primary gains**. Each candidate has its own selected workload/placement seed and its own matched BR time. Do not compare absolute times between different selected workloads as a causal policy comparison.','','| Cache | Batch | Policy | Candidates completed | Seed (workload, placement) | BR TTFT s | Policy TTFT s | Best gain % | All observed gain range % |','|---|---:|---|---:|---|---:|---:|---:|---|']
 for s in summaries:
  b=s['best'];p=b['policies'];lo,hi=s['gain_range_pct'];lines.append(f"| C{s['cache']} | {s['batch']} | {s['candidate']} | {s['n']}/8 | ({b['workload_seed']}, {b['placement_seed']}) | {p['BR']['TTFT_s']:.3f} | {p[s['candidate']]['TTFT_s']:.3f} | {b['gain_pct']:+.2f} | {lo:+.2f} to {hi:+.2f} |")
 lines+=['','## Work accounting and interpretation','','All completed pairs pass CPU physical-copy/state checks, cross-policy first-token parity, warm-to-measure token parity and no compilation during measurement. Each pair has equal per-rank expert H2D bytes/copies and equal total routed expert rows. No incomplete pair is used.','','For the same C30/B16 workload24 / placement20 pair:']
 b=next(x for x in pairs if x['key']=='C30_B16_CA_w24_p20');br=b['policies']['BR'];ca=b['policies']['CA']
 lines += [f"- BR -> CA TTFT: {br['TTFT_s']:.6f} -> {ca['TTFT_s']:.6f} s.",f"- Payload wire bytes: {br['wire_bytes']/2**30:.3f} -> {ca['wire_bytes']/2**30:.3f} GiB ({b['wire_change_pct']:+.2f}%). This counter excludes metadata collective traffic.",f"- Sum over layers of maximum-rank expert rows: {br['sum_layer_max_rank_rows']:,} -> {ca['sum_layer_max_rank_rows']:,} ({b['critical_rows_change_pct']:+.2f}%).",f"- Rank expert rows: BR {br['expert_rows_by_rank']}; CA {ca['expert_rows_by_rank']}.",f"- Total H2D identical at {sum(br['H2D_bytes_by_rank'])/2**30:.3f} GiB.",'','CA constrains the number of admitted experts per rank while optimizing locality; that does not balance token work per expert. The measured routing concentrates hot-expert work on a rank. Lower payload traffic together with higher maximum-rank work is consistent with compute imbalance offsetting communication savings. Old CA has a similar cold-prefill assignment pattern. LA balances token work and shows modest observed gains. This is a mechanism hypothesis supported by work counts, not a measured decomposition of the slowdown; controller cost, GEMM shapes and synchronization can also contribute.','','Cache30/60 both start empty and visit each layer once: current-layer experts require the same mandatory transfers. Larger capacity changes retained/victim state, not a current-layer cache-hit benefit. Do not generalize this result to repeated decode or warm-cache inference.','','## Decision','','No stable GO/STOP threshold is claimed because S2 was canceled. Among the observed favorable seed screens, LA is the most promising candidate; CA/OLD_CA did not demonstrate positive headroom in the completed observations. No production-policy change is justified by this selection-only dataset. C60/B128 remains explicitly partial. No new GPU experiments are queued. Owned resident model loads are restored only on0/1/4/5;2/3/6/7 remain untouched.','','Full pair-level metrics, audited raw receipt paths/hashes, all observed gain ranges and interrupted-pair inventory are in `TTFT_STOPPED_RESULTS.json`.','']
 (PACKET/'TTFT_STOPPED_RESULTS.md').write_text('\n'.join(lines))
 print(json.dumps(dict(coverage=coverage,complete_pairs=len(pairs),incomplete_pairs=len(incomplete)),indent=2))
if __name__=='__main__':main()
