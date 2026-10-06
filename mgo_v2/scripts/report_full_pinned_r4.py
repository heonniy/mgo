"""Audit the fixed two-repeat full-model source-path A/B without exclusions."""
import argparse,hashlib,json,shutil,statistics
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PACKET=P/'experiments/full_pinned_r4_20261006'
def main(root):
 state=json.loads((root/'status.json').read_text())
 assert state['status']=='PASS',state.get('error')
 result=json.loads((root/'result.json').read_text());assert result['status']=='PASS'
 measurements=[];correctness=[];stores=[]
 for rank in range(4):
  store=json.loads((root/f'pinned_store_rank{rank}.json').read_text());stores.append(store)
  assert store['status']=='PASS' and store['bytes']==54*2**30
  reference=None;tokens=None
  for mode in ['STAGED','FULL_PINNED']:
   row=json.loads((root/f'{mode}_correctness_rank{rank}.json').read_text());correctness.append(row)
   assert row['status']=='PASS' and row['direct_pinned']==(mode=='FULL_PINNED')
   if reference is None:reference=row['observed'];tokens=row['argmax_hash']
   else:assert row['observed']==reference and row['argmax_hash']==tokens
   for repeat in [1,2]:
    row=json.loads((root/f'{mode}_r{repeat}_measure_rank{rank}.json').read_text());measurements.append(row)
    assert row['status']=='PASS' and row['no_compile_in_measure'] and row['argmax_hash']==tokens
    assert row['scheduler_metrics']['copies']==reference['copies'] and row['scheduler_metrics']['bytes']==reference['bytes']
    assert abs(row['TTFT']+row['decode_wall']-row['E2E_wall'])<1e-8
 for mode in ['STAGED','FULL_PINNED']:
  for repeat in [1,2]:
   for metric in ['TTFT','TPOT','decode_wall','E2E_wall']:
    value=max(x[metric] for x in measurements if x['mode']==mode and x['repeat']==repeat)
    assert value==result['samples'][mode][repeat-1][metric]
 raw=PACKET/'raw';raw.mkdir(exist_ok=True)
 files=list(root.glob('*_measure_rank*.json'))+list(root.glob('*_correctness_rank*.json'))+list(root.glob('pinned_store_rank*.json'))+list(root.glob('graph_receipt_rank*.json'))+[root/'result.json',root/'status.json',root/'run.log']
 hashes={}
 for file in files:
  shutil.copy2(file,raw/file.name);hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
 (PACKET/'SOURCE_HASHES.json').write_text(json.dumps(dict(source_commit=state.get('source_commit'),raw_root=str(root),sha256=hashes),indent=2)+'\n')
 lines=['# R4 full-model staged versus full-pinned expert sources','',
 'R4 GPUs 0,1,4,5; C30; local B128; BR/P2/T2; H1b; BF16; optimized overlap. Frozen teacher-forced routing and existing requests. One prefill output plus 64 decode steps (65 output positions). One correctness pass and exactly two counterbalanced measurements per mode; no sample exclusions.', '',
 'All timings in seconds. TTFT is local E2E minus decode wall; TPOT uses the existing CUDA-event decode timer /64. Each reported metric independently takes max across ranks, so aggregate TTFT+decode wall need not exactly equal aggregate E2E. This is not the global outer-wall production main-table timing protocol.', '',
 '| Mode | Repeat | TTFT | TPOT | Decode wall | E2E |', '|---|---:|---:|---:|---:|---:|']
 for mode,rows in result['samples'].items():
  for repeat,row in enumerate(rows,1):lines.append(f"| {mode} | {repeat} | {row['TTFT']:.6f} | {row['TPOT']:.6f} | {row['decode_wall']:.6f} | {row['E2E_wall']:.6f} |")
 lines+=['','| Metric | Staged estimate | Full-pinned estimate | Reduction | Staged spread | Full-pinned spread |','|---|---:|---:|---:|---:|---:|']
 for metric in ['TTFT','TPOT','decode_wall','E2E_wall']:
  lines.append(f"| {metric} | {result['estimates']['STAGED'][metric]:.6f} | {result['estimates']['FULL_PINNED'][metric]:.6f} | {100*result['gains'][metric]:+.3f}% | {100*result['relative_spread']['STAGED'][metric]:.3f}% | {100*result['relative_spread']['FULL_PINNED'][metric]:.3f}% |")
 lines+=['','Two-sample median equals mean. Spread=(max-min)/mean; values above 5% are unstable, not headline-ready. All samples are preserved without a third or open-ended repeat.', '',
 '| Rank / GPU | Pinned GiB | Untimed initialization seconds | Peak HBM GiB | Process max RSS GiB |', '|---|---:|---:|---:|---:|']
 for r,gpu in enumerate([0,1,4,5]):
  rows=[x for x in measurements if x['rank']==r]
  lines.append(f"| {r} / {gpu} | 54 | {stores[r]['init_seconds']:.3f} | {max(x['peak_gpu_bytes'] for x in rows)/2**30:.3f} | {max(x['max_rss_bytes'] for x in rows)/2**30:.3f} |")
 lines+=['','Pinned memory totals 216 GiB. Both source stores stay alive in the same process; allocation/copy is outside timing. Peak HBM and max RSS are process lifetime high-water marks, not isolated per-mode footprints. RSS includes shared file-backed pages, so summing rank RSS overstates unique host consumption.', '',
 'H1b graph input/output/weighted-output buffers are persistent during inference, not startup-only scratch. Their recorded sizes span 28.6–36.2 GiB per rank (see graph receipts). The current implementation does not release them when timing begins; both source modes reuse the same graphs. Reducing this GPU memory requires a separate buffer/graph implementation change.', '',
 'All four ranks passed token equality, canonical cache/controller validation and physical copy/byte/transport equality across both modes and all repetitions. No compilation occurred in primary measurements. Source-path DMA retains the scheduler, prefetch priority and slot hazards; direct mode skips CPU staging-team materialization as specified by e7209506.', '',
 f"Elapsed supervised execution: {state['finished_unix']-state['started_unix']:.1f} seconds. Host available before/after: {state['host_available_before']/2**30:.1f}/{state['host_available_after']/2**30:.1f} GiB. Raw records include all 16 measured rank rows, eight correctness receipts, four pinned-store receipts and four graph receipts."]
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('PASS: all 16 measured rank rows and eight correctness receipts audited.')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/home/hwlee/mgo-results/full_pinned_r4_20261006'));main(parser.parse_args().root)
