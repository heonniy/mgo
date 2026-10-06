"""Audit and publish every requested single-shot R4 BR/Near measurement."""
import csv,hashlib,json,shutil
from pathlib import Path
ROOT=Path('/home/hwlee/mgo-results/r4_br_near_h0_20261007')
PACKET=Path(__file__).resolve().parents[1]/'experiments/r4_br_near_h0_20261007'
def main():
 status=json.loads((ROOT/'status.json').read_text());assert status['status']=='PASS' and status['metadata_preflight']=='PASS'
 manifest=json.loads((ROOT/'manifest.json').read_text());aggregate=[];ranks=[];hashes={};results=[];differences=0
 raw=PACKET/'raw';raw.mkdir(exist_ok=True)
 def archive(path,dest):
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
  hashes[str(dest.relative_to(PACKET))]=hashlib.sha256(path.read_bytes()).hexdigest()
 for spec in manifest:
  src=ROOT/spec['label'];result=json.loads((src/'result.json').read_text());assert result['status']=='PASS' and result['measured_repeats']==1;results.append(result)
  for policy in ['BR','LA_CA_NEAR']:
   measured=[]
   for rank in range(4):
    row=json.loads((src/f'{policy}_r1_measure_rank{rank}.json').read_text());warm=json.loads((src/f'{policy}_warm_rank{rank}.json').read_text())
    assert row['status']==row['validation']['status']=='PASS' and row['validation']['no_compile'] and row['validation']['tokens_match_warm']
    assert row['argmax_hash']==warm['argmax_hash'] and row['validation']['graph_buffer_bytes']==0
    assert row['finite_logits'] and len(row['per_step_s'])==32
    measured.append(row);ranks.append(dict(cell=spec['label'],batch=spec['batch'],context=spec['context'],policy=policy,rank=rank,TTFT=row['TTFT'],TPOT=row['TPOT'],E2E=row['E2E'],copies=row['validation']['scheduler']['copies'],canceled=row['validation']['scheduler']['canceled'],peak_gpu_bytes=row['peak_gpu_bytes'],max_rss_bytes=row['max_rss_bytes']))
   for metric in ['TTFT','TPOT','E2E']:assert max(r[metric] for r in measured)==result['samples'][policy][metric]
   aggregate.append(dict(cell=spec['label'],batch=spec['batch'],context=spec['context'],policy=policy,repeat=1,**result['samples'][policy]))
  differences+=sum(x['different_tokens'] for x in result['policy_token_differences'])
  for file in src.glob('*.json'):archive(file,raw/spec['label']/file.name)
 for file in [ROOT/'status.json',ROOT/'manifest.json',ROOT/'result.json',ROOT/'run.log',ROOT/'metadata_preflight.log',*ROOT.glob('pinned_store_rank*.json')]:archive(file,raw/file.name)
 assert len(aggregate)==8 and len(ranks)==32
 for name,rows in [('PRIMARY_SAMPLES.csv',aggregate),('RANK_SAMPLES.csv',ranks)]:
  with (PACKET/name).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 (PACKET/'SOURCE_HASHES.json').write_text(json.dumps(dict(executed_commit=status['source_commit'],raw_root=str(ROOT),sha256=hashes),indent=2)+'\n')
 lines=['# R4 BR/Near on selected H0 full-pinned runtime','',
 'Completed eight requested primary measurements: four workloads x BR/LA_CA_NEAR, one sample each after per-policy correctness/warmup. No samples excluded or additional primary repeats. This is a single-shot comparison; repeat stability and statistical significance are unmeasured.', '',
 'R4 GPUs0/1/4/5, C30 capacities [461,461,461,460], local B16/B64 (global64/256 requests), exactly256/512 prompt tokens, decode32 (33 output positions including prefill). Existing frozen BR-generated traces/teacher inputs and prior selected placement seeds are reused. H0/full-pinned, V3 overlap, P2/T2 prefetch, BF16, unique combine, async metadata. No H1b graphs, strict per-layer phase barriers, separate diagnostics or trace recapture. Prefill cache state continues into decode. Near keeps the existing LA-style predictor-based prefetch placement.', '',
 'Model loading and 216 GiB rank-private CPU pinned-store construction are outside timing. Each policy starts with a reset cache; model/source pool persist across cells. TTFT/TPOT/E2E use the same wall-clock generation method as the old strict decode32 measurements, independently maximized over ranks. TPOT is decode wall /32. Runtime overlap, prefetch and source path differ from the old strict experiment, so its gains are not directly attributable to source pinning alone.', '',
 '| Local batch | Input | Policy | TTFT s | TPOT s/token | E2E s |', '|---:|---:|---|---:|---:|---:|']
 for x in aggregate:lines.append(f"| {x['batch']} | {x['context']} | {x['policy']} | {x['TTFT']:.6f} | {x['TPOT']:.6f} | {x['E2E']:.6f} |")
 lines+=['','| Local batch | Input | Near TTFT reduction | Near TPOT reduction | Near E2E reduction |','|---:|---:|---:|---:|---:|']
 for r in results:lines.append(f"| {r['batch']} | {r['context']} | {100*r['gains']['TTFT']:+.3f}% | {100*r['gains']['TPOT']:+.3f}% | {100*r['gains']['E2E']:+.3f}% |")
 lines+=['',f'All32 measured rank receipts pass independent CPU state/role/controller/copy-bound validation, finite logits, within-policy warm token equality, and no compilation during primary measurement. BR-vs-Near token differences in warmup: {differences} (BF16 reduction-order comparison; details preserved). Copies/cancellations remain recorded for every rank.', '',
 'An initial attempt stopped before timing because the compact metadata path rejected batches smaller than W128. The repair permits small batches only with supplied frozen gate scores; live small-batch history still raises explicitly. Distributed metadata byte/route/gate parity and buffer-reuse tests passed at B16/B64/B128/B256 (64 checks per rank). Failure logs are retained separately.', '',
 'These workloads were previously selected to expose BR-adversarial prefill headroom. They are not a representative average-case policy benchmark. One sample per policy does not establish reproducible small gains. Peak GPU bytes are process-lifetime PyTorch allocated high-water marks, not per-cell NVML peaks; RSS includes shared mapped backing. CPU pinned cost remains54 GiB/rank.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='PASS',primary_rows=len(aggregate),rank_rows=len(ranks),cross_policy_token_differences=differences),indent=2))
if __name__=='__main__':main()
