"""Reconcile actual uninstrumented copies against frozen CPU proof, decode only."""
import json,hashlib
from pathlib import Path
from run_refactor_measure import PACKET
from policy_regime_paths import ROOT,PACKET,ENVIRONMENT
def main():
 sources={};rows=[]
 def read(p):
  raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
 for setting in ('C30','C60'):
  base=ROOT/setting;inputs=base/'inputs_B128_H64';capture=base/f'POLICY_REGIME_{setting}_B128_H64';result=read(capture/'result.json')
  assert result['status']=='PASS'
  for policy in ('BR','OLD_CA','FCA','LA_CA'):
   proof=read(inputs/f'{policy}_P2_proof.json');fetches=read(inputs/f'{policy}_fetches.json');prefill=[0]*4
   for layer in fetches[:48]:
    for rank,*_ in layer:prefill[rank]+=1
   repeats=[]
   for repeat in range(1,len(result['pairs'])+1):
    ranks=[]
    for rank in range(4):
     x=read(capture/f'{policy}_r{repeat}_measure_rank{rank}.json');m=x['scheduler_metrics']
     assert x['status']=='PASS' and x['controller_counters']==proof['counters']
     assert m['copies']+m['canceled']==proof['max_copy_counts'][rank] and m['bytes']==m['copies']*9437184
     ranks.append(dict(rank=rank,decode_copies=m['copies']-prefill[rank],decode_bytes=m['bytes']-prefill[rank]*9437184,canceled=m['canceled']))
    repeats.append(dict(repeat=repeat,ranks=ranks,decode_copies=sum(x['decode_copies'] for x in ranks),decode_bytes=sum(x['decode_bytes'] for x in ranks),canceled=sum(x['canceled'] for x in ranks)))
   c=proof['counters'];rows.append(dict(setting=setting,policy=policy,prefill_copies_by_rank=prefill,decode_mandatory_fetches=c['mandatory']-sum(prefill),prefetch_issued=c['issued'],prefetch_useful=c['useful'],prefetch_precision=c['useful']/c['issued'],physical_repeats=repeats))
 for setting in ('C30','C60'):
  group=[x for x in rows if x['setting']==setting];base=next(x for x in group if x['policy']=='BR')
  for row in group:
   row['decode_mandatory_vs_BR']=row['decode_mandatory_fetches']/base['decode_mandatory_fetches']-1
   for x,b in zip(row['physical_repeats'],base['physical_repeats']):x['decode_copies_vs_BR']=x['decode_copies']/b['decode_copies']-1
 out=PACKET/'POLICY_REGIME_WORKLOAD.json';out.write_text(json.dumps(dict(status='PASS',rows=rows,source_sha256=sources,scope='Global counts/bytes, not summed rank wall time. Actual copy counts from clean physical repeats; prefill subtracted using same-policy no-prefetch proof because prefill disables prediction. Canceled decode prefetch requests remain separate. Useful means used by next layer, not necessarily ready before compute.'),indent=2)+'\n')
 for x in rows:print(x['setting'],x['policy'],x['decode_mandatory_fetches'],[r['decode_copies'] for r in x['physical_repeats']])
if __name__=='__main__':main()
