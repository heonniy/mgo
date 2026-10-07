"""Serial guarded trace captures, then deterministic five-policy CPU replays."""
import os,json,subprocess,time,shutil
from pathlib import Path
from main_eviction_replay import run
P=Path(__file__).resolve().parents[1];PACK=P/'experiments/main_eviction_history_20261007';ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007');PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'
def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def report(results):
 rows=['# MAIN eviction history results','','R4/C30, input256, empty cache before prefill; prefill retained into256 decode forwards. Prefetch OFF. Local batches8/16/64; global32/64/256. LFU counts one use per distinct layer/step expert. Counts below exclude prefill.','', '|Batch|Policy|MAIN hit|Miss|Hit %|Eviction|Reload|','|---|---|---:|---:|---:|---:|---:|']
 for r in results:
  for s in r['policies']:
   d=s['decode'];rows.append(f"|{r['local_batch']}|{s['policy']}|{d['hit']}|{d['miss']}|{100*d['hit_rate']:.2f}|{d['eviction']}|{d['reload']}|")
 rows+=['','Reset deletes policy frequency/recency on eviction; cumulative preserves history across eviction and re-admission. Analytical reload history is always retained and cannot affect replacement. A hit is an active expert already resident before admission, not a prefetch or token-weighted metric. Eviction counts actual removed residents; reload counts a miss for any previously admitted expert, including prefill residents.','', 'Per-step CSVs include prefill at step0 and separate decode-only cumulative counters. Checkpoints1/8/16/32/64/128/256 are in each SUMMARY.json. LRU reset/cumulative equivalence and exact Gate capture/replay event counts plus final cache parity are required.','', 'BR algorithm/seed42 and total1843 MAIN slots are fixed; owner maps may differ because policy-dependent misses alter admissions. These are replay cache metrics conditional on the same frozen source trace per batch, not physical timing or quality measurements. No-prefetch source capture is exact-only BF16 with gate eviction.']
 (PACK/'RESULTS.md').write_text('\n'.join(rows)+'\n');write(PACK/'SUMMARY.json',dict(status='PASS' if len(results)==3 else 'RUNNING',results=results))
def main():
 results=[];state=dict(status='RUNNING',started=time.time());write(PACK/'STATUS.json',state)
 try:
  for b in (8,16,64):
   label=f'main_eviction_B{b}_L256_H256_v1';d=ROOT/label;state.update(batch=b,stage='capture');write(PACK/'STATUS.json',state)
   if not d.exists():
    cmd=[PY,str(P/'scripts/run_headline_job.py'),'--label',label,'--system','Ours','--worker','headline_ours_worker.py','--cell',f'R4_C30_B{b}_L256_O257','--repeats','1','--policy','BR','--prefill-optimized','--prefill-layout-fast','--legacy-decode-layout','--capture-eviction-trace','--workloads',str(PACK/'WORKLOADS.json')]
    subprocess.run(cmd,check=True)
   deadline=time.monotonic()+3700
   while True:
    status=json.loads((d/'status.json').read_text())
    if status['status']!='RUNNING':break
    if time.monotonic()>deadline:raise TimeoutError('existing capture supervisor did not finish')
    time.sleep(5)
   assert status['status']=='PASS',status.get('error')
   state.update(stage='CPU replay');write(PACK/'STATUS.json',state)
   result=run(d/'routing_trace.npz',PACK/f'B{b}');results.append(result)
   for name in ('trace_receipt.json','status.json','result.json'):shutil.copy2(d/name,PACK/f'B{b}'/name)
   report(results)
  state.update(status='PASS',stage='complete',finished=time.time());write(PACK/'STATUS.json',state)
 except BaseException as e:
  state.update(status='FAILED',error=repr(e),finished=time.time());write(PACK/'STATUS.json',state);raise
if __name__=='__main__':main()
