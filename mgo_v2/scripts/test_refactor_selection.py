"""Scientific selection guardrails: exclude noise and never tune with LA."""
import json,tempfile
from pathlib import Path
import select_refactor_prefetch as s

def main():
 with tempfile.TemporaryDirectory() as t:
  s.ROOT=Path(t)/'raw';s.PACKET=Path(t)/'packet';s.PACKET.mkdir()
  for b in [128,256]:
   out=s.ROOT/f'TEST_B{b}_H64';out.mkdir(parents=True)
   rows=[]
   for p,value,unstable in [(1,1.015,False),(2,.5,True),(4,1.,False)]:
    rows.append(dict(case=dict(label=f'P{p}_T1',P=p,trigger='T1',policy='BR'),samples=[dict(TPOT=value,E2E_wall=10*value)]*2,unstable=unstable))
   (out/'result.json').write_text(json.dumps(dict(status='PASS',results=rows)))
  s.select('TEST',64);r=json.loads((s.PACKET/'M13_SCREEN_SELECTION.json').read_text())
  assert r['chosen']['P']==1 and len(r['rejected'])==1
  assert [x['P'] for x in r['finalists']]==[1,4]
  bad=s.ROOT/'TEST_B128_H64/result.json';d=json.loads(bad.read_text());d['results'][0]['case']['policy']='LA';bad.write_text(json.dumps(d))
  try:s.select('TEST',64)
  except AssertionError:pass
  else:raise AssertionError('LA tuning must be rejected')
 print('PASS selector excludes unstable winner, prefers small near-optimal P, rejects LA')
if __name__=='__main__':main()
