"""Assemble unchanged T0/T1 and independently remeasured, corrected T2 rows."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004')
PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'

def load(stage,batch):
 folder=ROOT/f'{stage}_B{batch}_H64';path=folder/'result.json'
 raw=path.read_bytes();result=json.loads(raw);status=json.loads((folder/'status.json').read_text())
 assert status['status']==result['status']=='PASS'
 return result['results'],dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),source_sha=status['source_sha'])

def assemble(original,repair,target):
 assert len({original,repair,target})==3
 prepared=[]
 for batch in (128,256):
  old,old_source=load(original,batch);new,new_source=load(repair,batch)
  old_by={r['case']['label']:r for r in old};new_by={r['case']['label']:r for r in new}
  expected={f'P{p}_T{t}' for p in (1,2,4) for t in (0,1,2)}
  assert len(old)==9 and set(old_by)==expected
  assert len(new)==3 and set(new_by)=={f'P{p}_T2' for p in (1,2,4)}
  rows=[];provenance={}
  for label in sorted(expected):
   corrected=label.endswith('_T2');row=(new_by if corrected else old_by)[label];case=row['case']
   assert case['policy']=='BR' and case['horizon']==64 and case['partial_precision']=='bf16'
   assert label==f'P{case["P"]}_{case["trigger"]}' and len(row['samples'])>=2
   if corrected:assert case.get('t2_host_completion') is True,'repaired T2 host-completion receipt missing'
   rows.append(row);provenance[label]=new_source if corrected else old_source
  result=dict(status='PASS',assembly_only=True,results=rows,row_sources=provenance,
              interpretation='No new measurements here. Original T0/T1 rows are unchanged; T2 rows come exclusively from the physical completion repair. Original T2 samples remain archived and are never pooled.')
  out=ROOT/f'{target}_B{batch}_H64';assert not out.exists(),str(out)
  prepared.append((out,result))
 # Validate both source groups before creating either output.
 for out,result in prepared:
  out.mkdir();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 receipt=dict(status='PASS',assembly_only=True,original_stage=original,repair_stage=repair,target_stage=target,
              outputs=[dict(path=str(out/'result.json'),sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest(),row_sources=result['row_sources']) for out,result in prepared])
 (PACKET/'M13_SCREEN_ASSEMBLY.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--original',default='M13_BF16_SCREEN');p.add_argument('--repair',default='M13_T2_SYNC_REPAIR');p.add_argument('--target',default='M13_BF16_SCREEN_CORRECTED');a=p.parse_args()
 assemble(a.original,a.repair,a.target)
