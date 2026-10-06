"""Archive and report the two owner-authorized prefill optimization cells."""
from pathlib import Path
import hashlib,json,shutil,statistics as stats
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
PACKET=Path(__file__).resolve().parents[1]/'experiments/main_table_global_workload_20261006/prefill_optimized'

def read(p):return json.loads(p.read_text())
def metric(values):
 return dict(samples=values,mean=stats.mean(values),sample_std=stats.stdev(values),spread_percent=(max(values)-min(values))/stats.mean(values)*100)
def main():
 results=[]
 for cell,job,old,old_repeats in [
 ('B16/L256','ours_B16_L256_prefill_optimized1','ours_B16_L256_confirmation2',3),
 ('B64/L512','ours_B64_L512_prefill_optimized1','ours_B64_L512_owner_ttft_rerun1',1)]:
  raw=ROOT/job;status=read(raw/'status.json');assert status['status']=='PASS'
  dst=PACKET/job;dst.mkdir(exist_ok=True)
  archived={}
  for f in list(raw.glob('*.json'))+[raw/'run.log']:
   shutil.copy2(f,dst/f.name);h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==hashlib.sha256((dst/f.name).read_bytes()).hexdigest();archived[f.name]=h
  samples=[read(raw/f'repeat{i}.json') for i in (1,2)];assert all(s['status']=='PASS' for s in samples)
  row=dict(cell=cell,job=job,source_commit=status['source_commit'],metrics={m:metric([s[m] for s in samples]) for m in ('TTFT','TPOT','E2E')},archive_sha256=archived)
  checks=[];token_difference=first_difference=total_tokens=total_requests=repeat_difference=0
  for rank in range(4):
   validation=read(raw/f'prefill_validation_rank{rank}.json');assert validation['status']=='PASS' and validation['metadata_exact_checks']==48
   assert len(validation['per_layer_numerics'])==48;checks.extend(validation['per_layer_numerics'])
   a=read(ROOT/old/f'repeat1_rank{rank}.json');b=read(raw/f'repeat1_rank{rank}.json');c=read(raw/f'repeat2_rank{rank}.json')
   assert a['request_ids']==b['request_ids']==c['request_ids']
   for receipt in (b,c):
    assert receipt['validation']['status']=='PASS' and receipt['no_compile'] and receipt['expert_cache_start']=='empty' and receipt['prefill_optimized'] and receipt['output_tokens']==64
   for x,y,z in zip(a['tokens'],b['tokens'],c['tokens']):
    assert len(x)==len(y)==len(z)==64
    total_requests+=1;total_tokens+=64;first_difference+=x[0]!=y[0];token_difference+=sum(t!=u for t,u in zip(x,y));repeat_difference+=sum(t!=u for t,u in zip(y,z))
  row['validation']=dict(metadata_exact_checks=192,max_layer_relative_l2=max(x['relative_l2'] for x in checks),max_layer_absolute_error=max(x['max_abs'] for x in checks),baseline_job=old,first_token_differences=first_difference,total_requests=total_requests,all_token_differences=token_difference,total_output_tokens=total_tokens,optimized_repeat_token_differences=repeat_difference)
  prior=[read(ROOT/old/f'repeat{i}.json')['TTFT'] for i in range(1,old_repeats+1)]
  row['historical_TTFT']=dict(job=old,samples=prior,median=stats.median(prior),mean=stats.mean(prior),sample_std=stats.stdev(prior) if len(prior)>1 else None,observed_reduction_percent=(1-row['metrics']['TTFT']['mean']/stats.mean(prior))*100)
  resources=[json.loads(line) for line in (raw/'resources.jsonl').read_text().splitlines()]
  row['resources']=dict(min_host_available_GiB=min(x['host_available'] for x in resources)/2**30,max_observed_HBM_GiB=max(g['used_mib'] for x in resources for g in x['gpus'])/1024,restored=status['restored'])
  row['timing_stable']=all(m['spread_percent']<=5 for m in row['metrics'].values());results.append(row)
 (PACKET/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
 lines=['# Prefill tail metadata + fused transport results','','Both requested R4/C30 Near/H0/full-pinned cells completed on GPUs0/1/4/5. Input manifests, local/global batch, input lengths and output64 match prior measurements. No other policy, CPU/GPU allocation or H2D overlap change. Two measured repeats per cell after a separate numerical/metadata validation and full64-token warmup; reset expert/history state before measured batches.','','## Primary measurements','','Seconds; mean ± sample standard deviation (n=2). All samples retained; unstable means any metric range/mean exceeds5%.','','| Cell | TTFT | TPOT | E2E | Timing |','|---|---:|---:|---:|---|']
 for r in results:
  vals=[f"{r['metrics'][m]['mean']:.6f} ± {r['metrics'][m]['sample_std']:.6f}" for m in ('TTFT','TPOT','E2E')];lines.append('| '+r['cell']+' | '+' | '.join(vals)+' | '+('PASS' if r['timing_stable'] else 'UNSTABLE')+' |')
 lines+=['','## Every measured sample','','| Cell | Repeat | TTFT | TPOT | E2E |','|---|---:|---:|---:|---:|']
 for r in results:
  for i in range(2):lines.append('| '+r['cell']+f' | {i+1} | '+' | '.join(f"{r['metrics'][m]['samples'][i]:.6f}" for m in ('TTFT','TPOT','E2E'))+' |')
 lines+=['','## Historical comparison','','B16 reference is the mean of all three original final-confirmation samples; B64 reference is the subsequent owner-requested single-shot rerun. This is sequential historical comparison, not interleaved causal validation. Additional diagnostic prefill precedes the new warmup. Do not substitute selected historical samples, claim a stable speedup, or attribute the whole difference to either individual change.','','| Cell | Historical TTFT | New mean TTFT | Observed reduction |','|---|---:|---:|---:|']
 for r in results:lines.append(f"| {r['cell']} | {r['historical_TTFT']['mean']:.6f} | {r['metrics']['TTFT']['mean']:.6f} | {r['historical_TTFT']['observed_reduction_percent']:.2f}% |")
 lines+=['','## Validation and numerical semantics','','All192 layer/rank metadata comparisons per cell agree exactly with full gather, and all primary cache/copy consistency, finite-logit and no-compilation guards pass. CPU distributed regression passes7 cases including empty/unequal ranks. Default runtime selection regression passes. Initial CPU test invocation omitted the existing numba dependency path and failed before tests; rerun with the harness dependency path passed.','','Fused BF16 rank-partial accumulation changes association; it does not preserve bitwise expert-order outputs. Layer numerical comparisons use identical expert contributions, not an independent end-to-end baseline. Native autoregressive output differences versus the stated baseline repeat1 are reported below; they are not an accuracy evaluation and affect subsequent decode routes.','','| Cell | Max layer relative L2 | Max abs | First-token differences | All output differences | New repeats differ |','|---|---:|---:|---:|---:|---:|']
 for r in results:
  v=r['validation'];lines.append(f"| {r['cell']} | {100*v['max_layer_relative_l2']:.4f}% | {v['max_layer_absolute_error']:.6f} | {v['first_token_differences']}/{v['total_requests']} | {v['all_token_differences']}/{v['total_output_tokens']} | {v['optimized_repeat_token_differences']} |")
 lines+=['','## Implementation limits','','Only probability history payload is reduced to the exact global last128 rows. Selected expert IDs and routing weights still cover every token, as do CPU policy/layout work. The CPU probability array shrinks from64MiB to64KiB per layer/rank at B64 and8MiB to64KiB at B16. These are analytical array sizes, not measured wire bytes: padded all-gather still transmits up to128 rows per rank. Prefill uses packed forward plus BF16 rank-partial return; required-H2D wait/global barrier remains and prefill overlap stays disabled. This is not grouped-GEMM fusion. No default promotion or extra model runs.','','All receipts/logs are archived with hashes in RESULTS.json. Full resource series remain in the recorded raw job directories. Supervisors pass and restore owned idle model loads only on GPUs0/1/4/5.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps([{k:r[k] for k in ('cell','metrics','timing_stable','validation','historical_TTFT')} for r in results],indent=2))
if __name__=='__main__':main()
