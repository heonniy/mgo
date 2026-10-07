"""Archive BR single-shot timing and separate rank diagnostic summaries."""
import json,shutil,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parents[1]
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
PACK=P/'experiments/br_sync_audit_20261007'
def main():
 results=[]
 for batch in (16,64):
  d=ROOT/f'br_sync_B{batch}_L256_v1'
  status=json.loads((d/'status.json').read_text());assert status['status']=='PASS'
  primary=json.loads((d/'repeat1.json').read_text())
  out=PACK/f'B{batch}';out.mkdir(exist_ok=True)
  for name in ['status.json','result.json','repeat0.json','repeat1.json']+[f'repeat1_rank{r}.json' for r in range(4)]:shutil.copy2(d/name,out/name)
  ranks=[];cache_events=[]
  for rank in range(4):
   x=json.loads((d/f'generation_diagnostic_rank{rank}.json').read_text());assert x['token_and_cache_parity']
   cache_events.extend(x['decode_cache_events'])
   phases={mode:defaultdict(float) for mode in ('prefill','decode')};cpu={mode:defaultdict(float) for mode in phases}
   for s in x['segments']:
    mode='prefill' if s['event_index']<48 else 'decode'
    phases[mode][s['phase']]+=s['stream_seconds'];cpu[mode][s['phase']]+=s['cpu_ns']/1e9
   service=[c['service_seconds'] for c in x['h2d_copies']]
   ranks.append(dict(rank=rank,gpu=[0,1,4,5][rank],partition_seconds=phases,cpu_seconds=cpu,diagnostic_wall=x['wall_seconds'],h2d=dict(copies=len(service),GiB=sum(c['bytes'] for c in x['h2d_copies'])/2**30,service_seconds=sum(service),service_ms_percentiles=dict(zip(['p50','p90','p99','max'],[float(v) for v in np.percentile(service,[50,90,99,100])*1000]))),token_and_cache_parity=x['token_and_cache_parity']))
  def cache_summary(events):
   totals={k:sum(e[k] for e in events) for k in events[0] if k not in ('event_index','step','layer')}
   totals['avoid_demand_rate']=(totals['main_hits']+totals['prefetch_hits'])/totals['expert_uses']
   totals['main_hit_rate']=totals['main_hits']/totals['expert_uses']
   totals['token_weighted_avoid_demand_rate']=(totals['main_hit_tokens']+totals['prefetch_hit_tokens'])/totals['token_expert_uses']
   totals['surviving_prefill_hit_rate']=totals['surviving_prefill_hits']/totals['expert_uses']
   return totals
  cache=dict(first_step=cache_summary([e for e in cache_events if e['step']==1]),all_decode=cache_summary(cache_events),per_step={str(step):cache_summary([e for e in cache_events if e['step']==step]) for step in range(1,64)})
  assert cache['all_decode']['token_expert_uses']==batch*4*8*48*63
  primary_rank0=json.loads((d/'repeat1_rank0.json').read_text())
  assert cache['all_decode']['prefetch_hits']==primary_rank0['validation']['controller']['useful']
  row=dict(decode_cache=cache,batch=batch,input=256,output=64,primary=primary,ranks=ranks,raw=str(d),note='Single primary only. Separate diagnostic stream intervals include host gaps and peer waits; H2D service is non-additive.')
  (out/'DIAGNOSTIC_SUMMARY.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
 (PACK/'SUMMARY.json').write_text(json.dumps(results,indent=2)+'\n')
 files=['examples/headline_ours_worker.py','scripts/generation_phase_diagnostics.py','scripts/prefill_phase_diagnostics.py','mgo_v2/decode_runtime.py','mgo_v2/selected_runtime.py']
 (PACK/'SOURCE_HASHES.json').write_text(json.dumps({f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
 lines=['# BR current-runtime audit results', '', 'R4/C30, local B16 or B64 (global64/256), input256, output64, BR, H0/full-pinned, optimized prefill, original decode. One clean primary per cell; no stability estimate.', '', '|Local batch|TTFT s|TPOT s|E2E s|', '|---|---:|---:|---:|']
 for r in results:
  q=r['primary'];lines.append(f"|{r['batch']}|{q['TTFT']:.6f}|{q['TPOT']:.6f}|{q['E2E']:.6f}|")
 lines+=['', '## Decode cache reuse', '', 'Expert-use denominators deduplicate each expert within a layer/step/rank. Token-weighted denominators count every routed token-expert use. These are logical hits avoiding a new demand copy, not a claim of zero physical H2D wait.', '', '|Batch / window|MAIN hit|Prefetch hit|Avoid demand|Token-weighted avoid demand|Surviving prefill hit|', '|---|---:|---:|---:|---:|---:|']
 for r in results:
  for mode in ('first_step','all_decode'):
   q=r['decode_cache'][mode];lines.append(f"|B{r['batch']} / {mode}|{q['main_hit_rate']:.2%}|{q['prefetch_hits']/q['expert_uses']:.2%}|{q['avoid_demand_rate']:.2%}|{q['token_weighted_avoid_demand_rate']:.2%}|{q['surviving_prefill_hit_rate']:.2%}|")
 lines+=['', '## Separate diagnostic: per-rank mean milliseconds per decode token', '', 'Ranges below are minimum–maximum across four ranks. These completion spans include CPU submission gaps and peer waits; expert_compute is not isolated GPU GEMM time. H2D service overlaps and is non-additive. Diagnostic overhead can also change ready-first timing; do not substitute these spans for primary TPOT.', '', '|Phase|B16 ms/token|B64 ms/token|', '|---|---:|---:|']
 groups={'Metadata':['moe.metadata'],'Placement controller':['placement_controller_cpu'],'CPU layout':['layout_cpu'],'GPU index materialization':['layout_device_materialization'],'Dispatch submission + completion':['moe.forward_a2a','forward_token_a2a_submit','moe.forward_complete'],'Required H2D exposed dependency':['required_h2d_exposed_wait'],'Expert execution incl. host launches':['moe.expert_compute'],'Return collective':['return_token_a2a'],'Local partial / combine':['moe.return_a2a'],'Prefetch controller':['moe.prefetch_controller']}
 for name,keys in groups.items():
  vals=[]
  for r in results:
   v=[sum(rank['partition_seconds']['decode'].get(k,0) for k in keys)*1000/63 for rank in r['ranks']];vals.append(f'{min(v):.3f}–{max(v):.3f}')
  lines.append('|'+name+'|'+'|'.join(vals)+'|')
 lines+=['', 'See STRUCTURE.md for exact barrier and stream dependencies. Prefill has a required-H2D global barrier; overlapped decode does not. Neither selected path adds a post-expert global barrier. Collective completion can include slower-peer waiting.', '', 'All diagnostic tokens and final cache/role hashes match the corresponding primary. Primary receipts report no compilation. Full events remain in the raw paths recorded in SUMMARY.json; all requested measurements are preserved.', '', 'B16 launched at fb30aab; the diagnostic module was extended during warmup before its first lazy import (675896e and 3daa20a). Runtime/primary semantics were unchanged. SOURCE_HASHES.json records the executed diagnostic implementation. No additional run or optimization is authorized by this audit.']
 lines+=['', '## Findings and limits', '',
 '1. B64 return completion is rank-skew sensitive: GPU0 expert span572.48ms/token and return36.29ms, GPU1 expert497.02ms and return116.87ms. This is consistent with faster ranks waiting for the slower peer; it does not isolate pure network time or prove which upstream subcomponent causes all skew.',
 '2. Expert spans contain substantial host work: own-thread CPU time is377.59–412.87ms/token for B16 and488.10–538.81ms/token for B64. H0 launches experts individually; host readiness polling, gathers, launches and weighting remain. CPU layout plus GPU-index preparation also grows from about62–65ms to109–116ms/token. These are diagnostic costs, not a measured speedup opportunity of the same size.',
 '3. No per-expert required-H2D waits were observed during diagnostic decode. Copies still total1547.98GiB (B16) and2138.40GiB (B64) across prefill+decode, all ranks, including prefetch. Their service overlaps the critical stream. Diagnostic overhead gives transfers more time to finish, so this does not prove zero H2D impact in uninstrumented primary timing.',
 '4. H2D medians span0.181–0.200ms/copy in B16 and0.184–0.211ms in B64; p99 reaches0.765ms and0.711ms respectively. These samples show tails and rank variation but do not establish severe contention as the dominant bottleneck; no interference-controlled ablation was run.',
 '5. MAIN hits persist, but original prefill entries gradually turn over. Continuously surviving prefill entries account for13.26% of all decode expert uses in B16 and2.45% in B64. Total token-weighted no-new-demand coverage remains about76% in both. Prefetch hits avoid a new demand request but consumed earlier H2D; no-prefetch causal savings were not measured.',
 '6. Instrumented decode averages about0.857s/token and1.108s/token versus clean0.785s and1.022s. Diagnostic overhead is observable; never mix these timings. One primary per condition does not establish stability or a BR-versus-Near policy gain.',
 '7. Existing main-table OURS already used P2/T2 prefetch. A historical B16/L256 primary has23257 useful promotions. This audit changes only policy to BR and enables separate post-primary diagnostics; the default policy remains Near and the original decode remains default.'
 ]
 (PACK/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps([dict(batch=r['batch'],primary=r['primary']) for r in results],indent=2))
if __name__=='__main__':main()
