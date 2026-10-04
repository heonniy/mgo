"""BR-only frozen selector; never consumes LA timings."""
import json,math,statistics,argparse
from pathlib import Path
from ca_stress_common import write,sha
P=Path(__file__).resolve().parents[1];PACKET=P/'experiments/decode_prefetch_runtime_refactoring_20261004';ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
def select(stage,horizon):
 bybatch={};sources={}
 for batch in [128,256]:
  path=ROOT/f'{stage}_B{batch}_H{horizon}/result.json';d=json.loads(path.read_text());assert d['status']=='PASS';sources[str(batch)]=sha(path)
  for r in d['results']:assert r['case']['policy']=='BR'
  bybatch[batch]={r['case']['label']:r for r in d['results']}
 modes={r['case'].get('partial_precision','exact') for b in bybatch.values() for r in b.values()};assert len(modes)==1;precision=next(iter(modes))
 common=sorted(set(bybatch[128]) & set(bybatch[256]));eligible=[];rejected=[]
 for label in common:
  rows=[bybatch[b][label] for b in [128,256]]
  if any(r['case']['trigger']=='T2' and not r['case'].get('t2_host_completion',False) for r in rows):
   rejected.append(dict(label=label,reason='legacy T2 only ordered CUDA streams; host completion not established',rows=rows));continue
  if any(r['unstable'] for r in rows):rejected.append(dict(label=label,reason='unstable',rows=rows));continue
  tpots={str(b):bybatch[b][label].get('estimate',{}).get('TPOT',statistics.median(x['TPOT'] for x in bybatch[b][label]['samples'])) for b in [128,256]}
  eligible.append(dict(label=label,P=rows[0]['case']['P'],trigger=rows[0]['case']['trigger'],TPOT=tpots,rows=rows))
 assert eligible,'No stable shared candidate; concrete stability diagnosis is required'
 best={str(b):min(r['TPOT'][str(b)] for r in eligible) for b in [128,256]}
 for r in eligible:
  r['normalized']={b:r['TPOT'][b]/best[b] for b in best};r['score']=math.sqrt(math.prod(r['normalized'].values()))
 ranked=sorted(eligible,key=lambda r:(r['score'],r['P'],['T1','T2','T0'].index(r['trigger'])))
 close=[r for r in ranked if max(r['normalized'].values())<=1.02]
 winner=min(close,key=lambda r:(r['P'],['T1','T2','T0'].index(r['trigger']),r['score'])) if close else ranked[0]
 ordered=[winner]+[r for r in ranked if r['label']!=winner['label']]
 result=dict(status='PASS',stage=stage,horizon=horizon,common_return_precision=precision,source_hashes=sources,eligible=ranked,rejected=rejected,chosen=winner)
 if horizon==64:
  finalists=ordered[:2];result['finalists']=finalists;groups=[]
  for batch in [128,256]:
   cases=[] if precision=='bf16' else [dict(label='REFERENCE',policy='BR',P=0,trigger='T1',horizon=256,baseline=True,measure=False)]
   for r in finalists:cases.append(dict(label=r['label'],policy='BR',P=r['P'],trigger=r['trigger'],horizon=256,overlap=True,**({'partial_precision':precision} if precision!='exact' else {})))
   groups.append(dict(batch=batch,horizon=256,cases=cases))
  write(PACKET/'M13_CONFIRM_CONFIG.json',groups);write(PACKET/'M13_SCREEN_SELECTION.json',result)
 else:
  result.update(frozen=True,policy_used='BR only',immutable_for_LA=True);write(PACKET/'M13_FROZEN_PREFETCH.json',result)
 print(json.dumps(dict(stage=stage,chosen=winner['label'],TPOT=winner['TPOT'],eligible=len(eligible),rejected=len(rejected))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--horizon',type=int,choices=[64,256],required=True);a=p.parse_args();select(a.stage,a.horizon)
