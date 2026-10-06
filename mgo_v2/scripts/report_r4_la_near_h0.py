"""Audit the one-cell, two-policy clean LA/Near comparison."""
import csv,hashlib,json,shutil
from pathlib import Path
ROOT=Path('/home/hwlee/mgo-results/r4_la_near_h0_20261007')
PACKET=Path(__file__).resolve().parents[1]/'experiments/r4_la_near_h0_20261007'
def main():
 status=json.loads((ROOT/'status.json').read_text());assert status['status']=='PASS' and status['metadata_preflight']=='PASS'
 manifest=json.loads((ROOT/'manifest.json').read_text());assert len(manifest)==1
 spec=manifest[0];assert spec['batch']==64 and spec['context']==512
 src=ROOT/spec['label'];r=json.loads((src/'result.json').read_text());assert r['status']=='PASS' and r['measured_repeats']==1 and r['baseline_policy']=='LA' and r['candidate_policy']=='LA_CA_NEAR'
 rows=[];samples=[]
 for policy in ['LA','LA_CA_NEAR']:
  measured=[]
  for rank in range(4):
   row=json.loads((src/f'{policy}_r1_measure_rank{rank}.json').read_text());warm=json.loads((src/f'{policy}_warm_rank{rank}.json').read_text());v=row['validation']
   assert row['status']==v['status']=='PASS' and v['no_compile'] and v['tokens_match_warm'] and v['graph_buffer_bytes']==0
   assert row['argmax_hash']==warm['argmax_hash'] and row['finite_logits'] and len(row['per_step_s'])==32
   measured.append(row);rows.append(dict(policy=policy,rank=rank,TTFT_s=row['TTFT'],TPOT_s=row['TPOT'],E2E_s=row['E2E'],copies=v['scheduler']['copies'],canceled=v['scheduler']['canceled'],forward_bytes=v['forward_bytes'],return_bytes=v['return_bytes']))
  for m in ['TTFT','TPOT','E2E']:assert max(x[m] for x in measured)==r['samples'][policy][m]
  samples.append(dict(policy=policy,repeat=1,**r['samples'][policy]))
 for name,data in [('PRIMARY_SAMPLES.csv',samples),('RANK_SAMPLES.csv',rows)]:
  with (PACKET/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
 hashes={}
 for file in sorted(ROOT.rglob('*')):
  if file.is_file() and file.suffix in ['.json','.log']:
   dest=PACKET/'raw'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest);hashes[str(dest.relative_to(PACKET))]=hashlib.sha256(file.read_bytes()).hexdigest()
 (PACKET/'SOURCE_HASHES.json').write_text(json.dumps(dict(source_commit=status['source_commit'],sha256=hashes),indent=2)+'\n')
 diffs=sum(x['different_tokens'] for x in r['policy_token_differences']);count=sum(x['total_tokens'] for x in r['policy_token_differences'])
 lines=['# Pure LA versus LA_CA_NEAR: clean R4/B64/L512 timing','',
 'Completed exactly one clean primary measurement per policy, after one full warm correctness pass each. No phase instrumentation or additional timing repeats. Results are single-shot observations, not repeat-stability or significance evidence.','',
 'R4 GPUs 0,1,4,5; C30 capacities [461,461,461,460]; local B64 (global B256); input512; decode32 plus first output from prefill. H0/full-pinned V3 P2/T2, BF16, unique combine, async metadata. Same original frozen routes, teacher tokens and placement seed28. Policies apply to both prefill and decode, with prefill cache carried into decode. Pure LA is controller policy4; LA_CA_NEAR is policy7. Existing prefetch behavior is retained in both. No new trace capture, seed search or graph cache.','',
 'Model loading and 216 GiB CPU pinned-store construction are outside timing. Each policy starts with fresh cache state, reusing the model and source pool. Measurement order is LA then LA_CA_NEAR after both warmups. TTFT, TPOT and E2E are independently maximized over the four ranks; TPOT is decode wall time divided by32.','',
 '| Policy | TTFT s | TPOT ms/token | E2E s |','|---|---:|---:|---:|']
 for x in samples:lines.append(f"| {x['policy']} | {x['TTFT']:.6f} | {x['TPOT']*1000:.3f} | {x['E2E']:.6f} |")
 lines+=['',f"Near reduction relative to pure LA: TTFT {100*r['gains']['TTFT']:+.3f}%, TPOT {100*r['gains']['TPOT']:+.3f}%, E2E {100*r['gains']['E2E']:+.3f}%. Negative means Near was slower.",'',
 f'All8 measured rank receipts passed CPU state/role/controller/copy-bound checks, finite logits, within-policy warm/measurement token equality and no compilation during timing. Cross-policy output differences: {diffs}/{count}; frozen teacher inputs remain identical, and cross-policy bitwise equality is not claimed. No further numerical investigation was added, consistent with the owner BF16 scope.','',
 'Input hashes and pure-LA CPU replay were validated before launch; Near reused its existing independent reference. Distributed metadata parity preflight passed. Host memory guards remained enabled; no OOM. Owned model-forward loads restored only on0,1,4,5 after completion. GPUs2,3,6,7 were untouched.','',
 'The inherited workload/seed was selected for BR-adversarial prefill headroom. This is not an average-case LA-versus-Near benchmark. Previous BR/Near primary results and diagnostic timings are separate runs and are not used as this comparison\'s baseline. No timing samples were excluded.','',
 f"Executed source commit: `{status['source_commit']}`.",'',
 'Artifacts: [PRIMARY_SAMPLES.csv](PRIMARY_SAMPLES.csv), [RANK_SAMPLES.csv](RANK_SAMPLES.csv), [INPUTS_AND_CPU_PROOFS.json](INPUTS_AND_CPU_PROOFS.json), [SOURCE_HASHES.json](SOURCE_HASHES.json), and all warm/primary rank receipts in [raw/](raw/).']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='PASS',samples=samples,gains=r['gains'],cross_policy_token_differences=diffs),indent=2))
if __name__=='__main__':main()
