"""Never pool old/new T2, accept incomplete sources, or rewrite original rows."""
import json,tempfile
from pathlib import Path
import assemble_refactor_screen as a

def main():
 with tempfile.TemporaryDirectory() as tmp:
  a.ROOT=Path(tmp);a.PACKET=a.ROOT/'packet';a.PACKET.mkdir()
  for batch in (128,256):
   for stage,ts,value in [('OLD',(0,1,2),10),('FIX',(2,),20)]:
    out=a.ROOT/f'{stage}_B{batch}_H64';out.mkdir();rows=[]
    for p in (1,2,4):
     for t in ts:
      case=dict(label=f'P{p}_T{t}',P=p,trigger=f'T{t}',policy='BR',horizon=64,partial_precision='bf16')
      if stage=='FIX':case['t2_host_completion']=True
      rows.append(dict(case=case,samples=[dict(TPOT=value,E2E_wall=value)]*2,unstable=False))
    (out/'result.json').write_text(json.dumps(dict(status='PASS',results=rows)))
    (out/'status.json').write_text(json.dumps(dict(status='PASS',source_sha=stage)))
  source=a.ROOT/'OLD_B128_H64/result.json';before=source.read_bytes()
  a.assemble('OLD','FIX','JOIN')
  merged=json.loads((a.ROOT/'JOIN_B128_H64/result.json').read_text())
  assert source.read_bytes()==before
  for row in merged['results']:
   corrected=row['case']['trigger']=='T2'
   assert row['samples'][0]['TPOT']==(20 if corrected else 10)
   assert merged['row_sources'][row['case']['label']]['source_sha']==('FIX' if corrected else 'OLD')
  bad=a.ROOT/'FIX_B256_H64/result.json';x=json.loads(bad.read_text());x['results'][0]['case'].pop('t2_host_completion');bad.write_text(json.dumps(x))
  try:a.assemble('OLD','FIX','INVALID')
  except AssertionError:pass
  else:raise AssertionError('unmarked repair must fail')
  assert not (a.ROOT/'INVALID_B128_H64').exists()
 print('PASS source preservation, T2 replacement without pooling, fail-before-write for unverified repair')

if __name__=='__main__':main()
