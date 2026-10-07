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
  row=dict(decode_cache=cache,batch=batch,input=256,output=64,primary=primary,ranks=ranks,raw=str(d),note='Single primary only. Separate diagnostic stream intervals include host gaps and peer waits; H2D service is non-additive.')
  (out/'DIAGNOSTIC_SUMMARY.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
 (PACK/'SUMMARY.json').write_text(json.dumps(results,indent=2)+'\n')
 files=['examples/headline_ours_worker.py','scripts/generation_phase_diagnostics.py','scripts/prefill_phase_diagnostics.py','mgo_v2/decode_runtime.py','mgo_v2/selected_runtime.py']
 (PACK/'SOURCE_HASHES.json').write_text(json.dumps({f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
 print(json.dumps([dict(batch=r['batch'],primary=r['primary']) for r in results],indent=2))
if __name__=='__main__':main()
