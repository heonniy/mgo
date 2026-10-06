"""Audit and preserve all quick-fetch samples; no repeat exclusion."""
import csv,hashlib,json,shutil,statistics
from pathlib import Path
ROOT=Path('/home/hwlee/mgo-results/quick_fetch_path_20261006_retry2')
PACKET=Path(__file__).resolve().parents[1]/'experiments/quick_fetch_path_20261006'
def main():
 s=json.loads((ROOT/'summary.json').read_text());state=json.loads((ROOT/'status.json').read_text())
 assert s['status']==state['status']=='PASS' and s['repeats']==3
 rows=[]
 for rank,gpu in enumerate([0,1,4,5]):
  r=json.loads((ROOT/f'rank{rank}.json').read_text())
  assert r['rank']==rank and r['physical_gpu']==gpu and len(r['rows'])==72
  assert r['pinned_pool_bytes']==288*2**20
  rows.extend(r['rows'])
 assert len(rows)==288
 for item in s['summary']:
  for mode,values in item['modes'].items():
   selected=[r for r in rows if r['condition']==item['condition'] and r['burst']==item['burst'] and r['mode']==mode]
   critical=[max(r['wall_ms'] for r in selected if r['repeat']==rep) for rep in range(3)]
   assert statistics.median(critical)==values['critical_rank_wall_ms']['median']
   for row in selected:
    assert len(row['dma_ms'])==row['burst']
    assert len(row['stage_ms'])==(row['burst'] if mode=='current_staging' else 0)
 archive=PACKET/'raw';archive.mkdir(exist_ok=True)
 paths=[ROOT/'summary.json',ROOT/'status.json',ROOT/'run.log']+[ROOT/f'rank{r}.json' for r in range(4)]+list(ROOT.glob('nccl-*.log'))
 hashes={}
 for path in paths:
  shutil.copy2(path,archive/path.name);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 (PACKET/'SOURCE_HASHES.json').write_text(json.dumps(dict(executed_commit='bfa40e0',raw_root=str(ROOT),sha256=hashes),indent=2)+'\n')
 with (PACKET/'ALL_REPEATS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['condition','burst','mode','repeat','rank','physical_gpu','wall_ms'],extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
 lines=['# R4 expert-fetch path results', '',
 'Completed on GPUs 0,1,4,5 with three measured repeats after one warmup. 28 condition/burst combinations, two paths, 168 measured synchronized bursts and 288 active-rank samples. All samples retained. Successful attempt took %.2f seconds including initialization.'%(state['finished_unix']-state['started_unix']), '',
 'First attempt failed before measurement because the runner omitted the existing numba dependency path. Second attempt was stopped by the agent after initialization stalled; its generic owner STOP receipt is an operational cancellation, not a new user instruction. Restoring the previously established NCCL_CUMEM_ENABLE=0 IPC baseline allowed completion. The stall mechanism was not isolated. All failed receipts and raw directories are preserved.', '',
 '## Critical active-rank wall time', '',
 'Each repeat takes the maximum over active ranks. Values below are median [min, max] milliseconds. Spread is (max-min)/mean, flagged when >5%; no samples were discarded or extra repeats added.', '',
 '| Condition | Experts | Current staging ms | Direct pinned ms | Reduction | >5% spread |',
 '|---|---:|---:|---:|---:|---|']
 for item in s['summary']:
  vals=[];noisy=[]
  for mode,v in item['modes'].items():
   q=v['critical_rank_wall_ms'];vals.append(f"{q['median']:.3f} [{q['min']:.3f}, {q['max']:.3f}]")
   if (q['max']-q['min'])/q['mean']>.05:noisy.append(mode)
  lines.append(f"| {item['condition']} | {item['burst']} | {vals[0]} | {vals[1]} | {100*item['direct_pinned_gain']:.2f}% | {', '.join(noisy) or 'none'} |")
 lines += ['', '## Per-rank 32-expert comparison', '', '| GPU | Isolated current ms | All4 current ms | Slowdown | Isolated direct ms | All4 direct ms |', '|---:|---:|---:|---:|---:|---:|']
 all4=next(x for x in s['summary'] if x['condition']=='all4' and x['burst']==32)
 for rank,gpu in enumerate([0,1,4,5]):
  iso=next(x for x in s['summary'] if x['condition']==f'iso{rank}' and x['burst']==32)
  val=lambda x,m:x['modes'][m]['by_rank'][str(rank)]['wall_ms']['median']
  a,b=val(iso,'current_staging'),val(all4,'current_staging')
  lines.append(f"| {gpu} | {a:.3f} | {b:.3f} | {(b/a-1)*100:.1f}% | {val(iso,'direct_pinned'):.3f} | {val(all4,'direct_pinned'):.3f} |")
 lines += ['', '## Interpretation and limits', '',
 'All4/32 current-path rank medians cluster at 35.21–35.67 ms: no single rank dominates this burst. Relative to isolated ranks, current-path slowdown is about 29–46%. Direct-pinned all4 critical latency is 6.074 ms versus 35.669 ms current staging (82.97% reduction). The direction supports shared staging/host-path overhead, but does not identify PCIe saturation versus CPU memory bandwidth versus scheduling as a unique cause.', '',
 'All4/32 per-expert staging medians are 0.952–0.998 ms; current DMA medians 0.211–0.359 ms, versus direct DMA 0.177–0.181 ms. Queue delay and summed stage/DMA durations overlap and must not be added as independent wall-time components.', '',
 'Direct pinned is an upper-bound microbenchmark with only 288 MiB pinned per rank, not a full 54 GiB expert-store implementation. Direct mode also bypasses the scheduler/queue and uses a different submission loop, so the entire gain cannot be attributed only to the staging memcpy. Current-path profiling is enabled. The scheduler is created before timing for each burst; full-model persistent-worker behavior is not reproduced. No model forward, TTFT, TPOT, output validation or full-store pinned-memory feasibility was measured.', '',
 'All ranks access the same selected expert keys; real policy-dependent expert sets and sustained multi-layer traffic may differ. Execution PASS verifies completion and trace counts, not copied-payload equality. CPU affinity and source keys are preserved in rank receipts. The 32-expert all4 critical spread is 1.83% current and 0.46% direct; some smaller bursts have >5% spread and are flagged in the table.', '',
 'The >=10% microbenchmark criterion is met: a registered/pinned source path warrants a separate implementation experiment. This is not evidence of an 83% end-to-end inference gain. No full main-table or R8 run was launched. Owned model-forward load was restored only on GPUs 0,1,4,5 after measurement.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('Audited all 288 active-rank samples and 56 three-repeat path/condition groups.')
if __name__=='__main__':main()
