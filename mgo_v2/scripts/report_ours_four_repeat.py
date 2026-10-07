"""Report every sample of the owner-requested fixed four-repeat packet."""
import json,statistics as st,shutil,hashlib
from pathlib import Path
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
PACKET=Path(__file__).resolve().parents[1]/'experiments/main_table_global_workload_20261006/ours_four_repeat'
def read(p):return json.loads(p.read_text())
def main():
 results=[]
 for cell in ['B16_L256','B64_L512']:
  raw=ROOT/('ours_'+cell+'_four_repeat1');ref=ROOT/('ours_'+cell+'_layout_repair1')
  status=read(raw/'status.json');assert status['status']=='PASS' and status['worker_returncode']==0
  result=read(raw/'result.json');assert result['primary_repeats']==4 and result['headline_eligible']
  samples=[read(raw/f'repeat{i}.json') for i in range(1,5)]
  assert all(x['status']=='PASS' and x['output_tokens']==64 for x in samples)
  metrics={}
  for m in ['TTFT','TPOT','E2E','throughput']:
   values=[r[m] for r in samples];metrics[m]=dict(samples=values,mean=st.mean(values),sample_std=st.stdev(values),median=st.median(values),min=min(values),max=max(values),spread_percent=(max(values)-min(values))/st.mean(values)*100)
  tokens=0;state_matches=[]
  for rank in range(4):
   validation=read(raw/f'prefill_validation_rank{rank}.json');assert validation['layout_exact_checks']==validation['metadata_exact_checks']==48
   previous=read(ref/f'repeat1_rank{rank}.json')
   for repeat in range(1,5):
    row=read(raw/f'repeat{repeat}_rank{rank}.json')
    assert row['no_compile'] and row['finite_logits'] and row['expert_cache_start']=='empty' and row['prefill_optimized'] and row['prefill_layout_fast'] and row['validation']['status']=='PASS'
    assert row['tokens']==previous['tokens'] and row['request_ids']==previous['request_ids']
    tokens+=sum(map(len,row['tokens']))
    state_matches.append(dict(rank=rank,repeat=repeat,state=row['validation']['state_hash']==previous['validation']['state_hash'],roles=row['validation']['role_hash']==previous['validation']['role_hash'],controller=row['validation']['controller']==previous['validation']['controller']))
  out=PACKET/raw.name;out.mkdir(exist_ok=True);hashes={}
  for f in list(raw.glob('*.json'))+[raw/'run.log']:
   shutil.copy2(f,out/f.name);h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==hashlib.sha256((out/f.name).read_bytes()).hexdigest();hashes[f.name]=h
  resources=[json.loads(x) for x in (raw/'resources.jsonl').read_text().splitlines()]
  results.append(dict(cell=cell,job=str(raw),source_commit=status['source_commit'],metrics=metrics,samples=samples,token_positions_identical=tokens,reference_job=str(ref),state_matches=state_matches,stable_all_metrics=all(metrics[m]['spread_percent']<=5 for m in ['TTFT','TPOT','E2E']),min_host_available_GiB=min(r['host_available'] for r in resources)/2**30,max_observed_HBM_GiB=max(g['used_mib'] for r in resources for g in r['gpus'])/1024,restored=status['restored'],archive_sha256=hashes))
 (PACKET/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
 lines=['# Four-repeat OURS TTFT / TPOT / E2E','','Owner-requested exactly4 clean repeats per cell. R4/C30 on GPUs0/1/4/5, Near/H0/full-pinned, prefill-optimized and prefill-layout-fast, output64. Same frozen ShareGPT-long inputs; local B16/global64 with input256 and local B64/global256 with input512. No runtime changes beyond allowing a4-repeat count. No extra fifth repeat or outlier deletion.','','Preparation: one-token index/metadata validation on disjoint warmup inputs, full64-token disjoint warmup, then4 measured batches with expert/cache/history reset before each. Compiled kernels and CPU pinned backing retained. No post-primary diagnosis/profiler. All common-release/max-rank timing rules unchanged. TPOT=(E2E−TTFT)/63.','','## All measured samples','','Seconds; first repeat is retained.','','| Cell | Repeat | TTFT | TPOT | E2E |','|---|---:|---:|---:|---:|']
 for r in results:
  for i,s in enumerate(r['samples'],1):lines.append(f"| {r['cell']} | {i} | {s['TTFT']:.6f} | {s['TPOT']:.6f} | {s['E2E']:.6f} |")
 lines+=['','## Mean ± sample standard deviation','','n=4, denominator n−1. All4 samples contribute independently to every metric.','','| Cell | TTFT | TPOT | E2E |','|---|---:|---:|---:|']
 for r in results:lines.append('| '+r['cell']+' | '+' | '.join(f"{r['metrics'][m]['mean']:.6f} ± {r['metrics'][m]['sample_std']:.6f}" for m in ['TTFT','TPOT','E2E'])+' |')
 lines+=['','## Range and timing spread','','Spread=(max−min)/mean. A whole row is timing-stable only if all three spreads are at most5%; valid execution does not imply stable timing.','','| Cell | Metric | Min | Max | Median | Spread |','|---|---|---:|---:|---:|---:|']
 for r in results:
  for name in ['TTFT','TPOT','E2E']:
   m=r['metrics'][name];lines.append(f"| {r['cell']} | {name} | {m['min']:.6f} | {m['max']:.6f} | {m['median']:.6f} | {m['spread_percent']:.2f}% |")
 lines+=['','## Verification and limits','','All GPU index/metadata checks, finite logits, cache consistency and no-compilation guards pass. Every generated token is compared against the preceding layout-repaired primary on identical request IDs. The unchanged runtime has no intended policy or arithmetic change.','']
 for r in results:
  equal=all(all(x[k] for k in ['state','roles','controller']) for x in r['state_matches'])
  lines.append(f"- {r['cell']}: {r['token_positions_identical']} token positions identical; final cache roles/state/controller equality={equal}; whole-row timing stable={r['stable_all_metrics']}; minimum host available={r['min_host_available_GiB']:.2f}GiB.")
 lines+=['','Repeats run within one loaded process per cell; they do not estimate variation across independent process launches or hosts. Host/model warm state beyond the reset expert cache remains. This packet does not diagnose the cause of first-repeat differences. Earlier results are preserved and not pooled into these statistics. Source commits and archive checksums are in RESULTS.json; full resource logs remain in the raw directories. Owned model loads restored on0/1/4/5 after completion.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
