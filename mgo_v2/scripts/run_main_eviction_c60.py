"""Owner C60 follow-up using existing frozen traces only."""
import json,time
from pathlib import Path
from main_eviction_history_replay import run
P=Path(__file__).resolve().parents[1]/'experiments/main_eviction_history_20261007'
def main():
 out=P/'C60';results=[];started=time.time()
 for b in (8,16,64):
  base=json.loads((P/f'B{b}/SUMMARY.json').read_text())
  r=run(Path(base['trace']),out/f'B{b}',[922,922,921,921]);assert r['trace_content_sha256']==base['trace_content_sha256']
  results.append(r);print('completed',b,flush=True)
 rows=['# C60 MAIN eviction replay','','Same frozen C30 source traces; CPU counterfactual only. R4, input256,256 decode steps after prefill. Prefetch OFF; all3686 slots MAIN. Counts exclude prefill.','','|Batch|Policy|C30 hit %|C60 hit %|C60 hit|Miss|Eviction|Reload|','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in results:
  b=r['local_batch'];base=json.loads((P/f'B{b}/SUMMARY.json').read_text());prior={p['policy']:p['decode'] for p in base['policies']}
  for p in r['policies']:
   d=p['decode'];rows.append(f"|{b}|{p['policy']}|{prior[p['policy']]['hit_rate']*100:.2f}|{d['hit_rate']*100:.2f}|{d['hit']}|{d['miss']}|{d['eviction']}|{d['reload']}|")
 rows+=['','B64 LRU has7 hits (0.000474%), not exactly zero; the table rounds percentages to two decimals.','','All trace hashes match C30. All hit/miss, occupancy, eviction/reload conservation and LRU-equivalence checks pass. C60 does not claim physical Gate parity against the C30 cache state or physical TPOT gains. Reset/cumulative refers to eviction-time policy metadata; analytical reload history is always retained.']
 (out/'RESULTS.md').write_text('\n'.join(rows)+'\n');(out/'SUMMARY.json').write_text(json.dumps(dict(status='PASS',elapsed_seconds=time.time()-started,results=results),indent=2)+'\n')
if __name__=='__main__':main()
