"""Validate and archive exact-layout repair performance and component evidence."""
from pathlib import Path
import json,statistics as st,shutil,hashlib
from collections import defaultdict
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
PACKET=Path(__file__).resolve().parents[1]/'experiments/main_table_global_workload_20261006/prefill_layout_repair'
def read(p):return json.loads(p.read_text())
def stats(x):return dict(samples=x,mean=st.mean(x),sample_std=st.stdev(x),spread_percent=(max(x)-min(x))/st.mean(x)*100)
def main():
 results=[]
 for cell in ['B16_L256','B64_L512']:
  label='ours_'+cell+'_layout_repair1';raw=ROOT/label;old=ROOT/('ours_'+cell+'_prefill_optimized1');old_diag=ROOT/('ours_'+cell+'_prefill_diagnosis1')
  status=read(raw/'status.json');assert status['status']=='PASS'
  dst=PACKET/label;dst.mkdir(exist_ok=True);hashes={}
  for f in list(raw.glob('*.json'))+[raw/'run.log']:
   shutil.copy2(f,dst/f.name);h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==hashlib.sha256((dst/f.name).read_bytes()).hexdigest();hashes[f.name]=h
  timings={m:stats([read(raw/f'repeat{i}.json')[m] for i in (1,2)]) for m in ('TTFT','TPOT','E2E')}
  prior={m:stats([read(old/f'repeat{i}.json')[m] for i in (1,2)]) for m in ('TTFT','TPOT','E2E')}
  token_positions=0;state_checks=[];diagnostics=[];old_diagnostics=[]
  for rank in range(4):
   v=read(raw/f'prefill_validation_rank{rank}.json');assert v['status']=='PASS' and v['layout_exact_checks']==v['metadata_exact_checks']==48
   for repeat in (1,2):
    a=read(raw/f'repeat{repeat}_rank{rank}.json');b=read(old/f'repeat{repeat}_rank{rank}.json')
    assert a['no_compile'] and a['validation']['status']=='PASS' and a['expert_cache_start']=='empty' and a['prefill_layout_fast']
    assert a['request_ids']==b['request_ids'] and a['tokens']==b['tokens'];token_positions+=sum(map(len,a['tokens']))
    state_checks.append(dict(rank=rank,repeat=repeat,state_hash_equal=a['validation']['state_hash']==b['validation']['state_hash'],role_hash_equal=a['validation']['role_hash']==b['validation']['role_hash'],controller_equal=a['validation']['controller']==b['validation']['controller']))
   d=read(raw/f'post_diagnostic_rank{rank}.json');assert d['status']=='PASS' and d['first_token_parity'] and d['no_compile'] and d['validation']['status']=='PASS'
   assert abs(sum(d['exclusive_stream_partition'].values())-d['wall_seconds'])<1e-6
   diagnostics.append(d);old_diagnostics.append(read(old_diag/f'diagnostic_rank{rank}.json'))
  components={}
  for name in ['layout_cpu','layout_device_materialization','placement_controller_cpu','moe_other','moe.current_controller','moe.expert_compute','required_h2d_exposed_wait','attention_dense_residual']:
   prev=st.mean(d['exclusive_stream_partition'].get(name,0) for d in old_diagnostics);now=st.mean(d['exclusive_stream_partition'].get(name,0) for d in diagnostics)
   cpu=st.mean(sum(s['cpu_ns'] for s in d['segments'] if s['phase']==name)/1e9 for d in diagnostics)
   components[name]=dict(old_rank_mean_seconds=prev,new_rank_mean_seconds=now,new_own_thread_cpu_seconds=cpu)
  resource=[json.loads(l) for l in (raw/'resources.jsonl').read_text().splitlines()]
  results.append(dict(cell=cell,job=label,source_commit=status['source_commit'],timings=timings,prior_timings=prior,observed_TTFT_reduction_percent=(1-timings['TTFT']['mean']/prior['TTFT']['mean'])*100,exact_layout_layer_rank_checks=192,matching_primary_token_positions=token_positions,state_comparisons=state_checks,diagnostic_components=components,diagnostic_global_TTFT=max(d['wall_seconds'] for d in diagnostics),archive_sha256=hashes,min_host_available_GiB=min(x['host_available'] for x in resource)/2**30,max_observed_HBM_GiB=max(g['used_mib'] for x in resource for g in x['gpus'])/1024,restored=status['restored']))
 (PACKET/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
 lines=['# CPU prefill layout repair results','','R4/C30, Near/H0/full-pinned, prefill probability-tail + BF16 rank-partial transport. Same inputs, cache budget, placement policy, BF16 accumulation order and decode implementation. Physical GPUs0/1/4/5 only. Two unprofiled full64-output primary repeats per cell after validation and warmup; dynamic expert state reset before every batch. No sample filtering or extra repetitions.','','## Primary timings','','Mean ± sample SD, n=2; seconds. These are clean timings, excluding the subsequent one-token diagnostic.','','| Cell | TTFT | TPOT | E2E | TTFT samples |','|---|---:|---:|---:|---|']
 for r in results:
  vals=[f"{r['timings'][m]['mean']:.6f} ± {r['timings'][m]['sample_std']:.6f}" for m in ('TTFT','TPOT','E2E')];lines.append('| '+r['cell']+' | '+' | '.join(vals)+' | '+ ' / '.join(f'{v:.6f}' for v in r['timings']['TTFT']['samples'])+' |')
 lines+=['','## Historical before/after','','Same two-repeat protocol, sequential runs rather than an interleaved A/B. Old reference is the prior prefill-optimized implementation, not the original unfused baseline or owner-selected subset. Report timing spread rather than claiming stable speedup.','','| Cell | Old mean TTFT | New mean TTFT | Observed reduction | New TTFT spread |','|---|---:|---:|---:|---:|']
 for r in results:lines.append(f"| {r['cell']} | {r['prior_timings']['TTFT']['mean']:.6f} | {r['timings']['TTFT']['mean']:.6f} | {r['observed_TTFT_reduction_percent']:.2f}% | {r['timings']['TTFT']['spread_percent']:.2f}% |")
 lines+=['','## Targeted diagnostic evidence','','Seconds, four-rank mean service spans. CUDA intervals include host submission gaps; own-thread CPU is also preserved in RESULTS.json. Old diagnostic followed one-token warmup; new diagnostic follows full-generation primaries. The preparation difference prevents attributing unrelated component movements entirely to this repair. H2D service is not added into the additive timeline.','','| Cell | Component | Before | After |','|---|---|---:|---:|']
 for r in results:
  for name,v in r['diagnostic_components'].items():lines.append(f"| {r['cell']} | {name} | {v['old_rank_mean_seconds']:.6f} | {v['new_rank_mean_seconds']:.6f} |")
 lines+=['','## Exactness and implementation','','28 independent CPU layout/device-pack comparisons pass: empty, uneven, zero-active, variable effective lengths, single-owner and both full prefill sizes across four ranks. Numba compilation was excluded from the CPU timing observations and warmed before physical primaries. Each physical cell passes192 exact old/new GPU index comparisons (48 layers x4 ranks), including canonical peer/token/expert ordering, send/receive counts and expert row/column order. Metadata remains exact. All finite-logit/cache/no-Torch-recompilation guards pass.','','| Cell | Identical generated token positions, both primaries | End-state/role/controller equality |','|---|---:|---|']
 for r in results:
  eq=all(all(x[k] for k in ['state_hash_equal','role_hash_equal','controller_equal']) for x in r['state_comparisons']);lines.append(f"| {r['cell']} | {r['matching_primary_token_positions']} | {eq} |")
 lines+=['','No numerical-association change is introduced by this layout repair. Exact output parity is against the preceding BF16 rank-partial version; that version still differs from the original expert-order return.','','The new Numba builder walks rank-contiguous tokens, produces canonical peer/token/expert packets with bounded eight-element ordering, and builds stable expert groups by counting. NumPy arrays are retained through one contiguous device transfer. Unused exact-return arrays and duplicate selected-ID copies are omitted. Legacy paths remain available; enable `--prefill-optimized --prefill-layout-fast`. The Numba helper uses the existing dependency, not a new package.','','## Remaining opportunities','','Placement controller still performs token-level generic substitution/accounting passes with substitution disabled; its measured residual cost is a bounded next target. Potential repairs include specialized exact no-substitution preparation, fewer token scans and avoiding unused temporary arrays while preserving slot/eviction decisions and accounting. These have been inspected but are not implemented in this change. Miscellaneous host work partly disappears with the index objects; attention/dense and required communication must not be treated as removable overhead. No claim that first-repeat TTFT variability is fixed.','','All raw receipts/diagnostic segments and archive hashes are retained. Resource samples remain in the raw job directories. Owned model loads restored on0/1/4/5; no other GPUs touched.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
