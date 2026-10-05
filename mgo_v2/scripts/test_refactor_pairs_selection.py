"""Selection failures that could otherwise manufacture a winning LA gain."""
from copy import deepcopy
from prepare_refactor_arms import groups_for,ARMS
from summarize_refactor_pairs import evaluate

def main():
 selection=dict(status='PASS',frozen=True,policy_used='BR only',immutable_for_LA=True,horizon=256,common_return_precision='bf16',chosen=dict(P=2,trigger='T1'))
 records={a:{} for a in ARMS}
 # V1 has 20% relative gain but is absolutely slower than V2's 10% gain.
 for group in groups_for(selection,256,[128,256]):
  arm=group['runtime_arm'];br,la={ARMS[0]:(2,1.6),ARMS[1]:(1,.9),ARMS[2]:(1.05,1)}[arm]
  pairs=[dict(BR=dict(TPOT=br,E2E_wall=br*10),LA=dict(TPOT=la,E2E_wall=la*10)) for _ in range(2)]
  records[arm][group['batch']]=dict(status='PASS',baseline='BR',candidate='LA',physical_arena_count=1,cases=group['cases'],pairs=pairs)
 short=deepcopy(records)
 for bybatch in short.values():
  for row in bybatch.values():
   for case in row['cases']:case['horizon']=64
 assert evaluate(short,horizon=64,world=4)['world']==4
 out=evaluate(records);assert out['winner']==ARMS[1];assert out['rows'][0]['dominated_by']
 # An apparently fast candidate with unresolved noise cannot win.
 noisy=deepcopy(records)
 for b in (128,256):
  noisy[ARMS[1]][b]['pairs']=[dict(BR=dict(TPOT=1,E2E_wall=10),LA=dict(TPOT=x,E2E_wall=x*10)) for x in [.3,.9,.4]]
 out=evaluate(noisy);assert not out['rows'][1]['eligible']
 # Ranking and a positive improvement claim are distinct requirements.
 bad=deepcopy(records)
 for bybatch in bad.values():
  for row in bybatch.values():
   for pair in row['pairs']:pair['LA']={k:v*1.1 for k,v in pair['BR'].items()}
 outcome=evaluate(bad);assert outcome['winner'] is not None and not outcome['improvement_supported']
 changed=deepcopy(records);changed[ARMS[2]][256]['cases'][1]['P']=4
 try:evaluate(changed)
 except AssertionError:pass
 else:raise AssertionError('different BR/LA physical capacity must be rejected')
 print('PASS gain-only domination, unresolved noise, regression and mismatched capacity')
if __name__=='__main__':main()
