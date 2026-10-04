#!/usr/bin/env python3
"""Summarize Env1/Env2 four-GPU replica microbench into CPU-model calibration."""
import argparse,json,statistics,hashlib
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()

def stat(rows,kind,rows_value=None):
 vals=[]
 for r in rows:
  for x in r['records']:
   if x.get('kind')!=kind:continue
   if rows_value is not None and x.get('rows')!=rows_value:continue
   if x.get('active',True) and 'median_ms' in x:vals.append(float(x['median_ms']))
 if not vals:raise KeyError((kind,rows_value))
 return max(vals)

def p90(rows,kind,rows_value=None):
 vals=[]
 for r in rows:
  for x in r['records']:
   if x.get('kind')!=kind:continue
   if rows_value is not None and x.get('rows')!=rows_value:continue
   if x.get('active',True) and 'p90_ms' in x:vals.append(float(x['p90_ms']))
 if not vals:raise KeyError((kind,rows_value))
 return max(vals)

def env(root,name):
 path=root/name
 status=json.loads((path/'status.json').read_text());assert status['status']=='PASS'
 ranks=[json.loads((path/f'rank{i}.json').read_text()) for i in range(4)]
 assert all(x['status']=='PASS' and x['environment']==name for x in ranks)
 activation={}
 for n in (1,8,16,32,64,128,256,512,1024,2048):
  activation[str(n)]=dict(oneway_median_ms=stat(ranks,'activation_oneway',n),
                          oneway_p90_ms=p90(ranks,'activation_oneway',n),
                          roundtrip_median_ms=stat(ranks,'activation_roundtrip',n),
                          roundtrip_p90_ms=p90(ranks,'activation_roundtrip',n))
 compute={}
 for n in (64,128,256,512,1024,2048):
  compute[str(n)]=dict(median_ms=stat(ranks,'expert_compute',n),
                       p90_ms=p90(ranks,'expert_compute',n))
 return dict(
  environment=name,
  raw_receipts=[dict(path=str(path/f'rank{i}.json'),sha256=sha(path/f'rank{i}.json')) for i in range(4)],
  transport_env=status.get('transport_env',{}),
  H2D_9MiB=dict(median_ms=stat(ranks,'h2d_4rank'),p90_ms=p90(ranks,'h2d_4rank')),
  D2D_9MiB_onepair=dict(median_ms=stat(ranks,'d2d_expert_onepair'),p90_ms=p90(ranks,'d2d_expert_onepair')),
  D2D_9MiB_twopair=dict(median_ms=stat(ranks,'d2d_expert_twopair'),p90_ms=p90(ranks,'d2d_expert_twopair')),
  resident_D2D_overlap_H2D=dict(median_ms=stat(ranks,'resident_d2d_overlap_h2d'),p90_ms=p90(ranks,'resident_d2d_overlap_h2d')),
  miss_overlap=dict(median_ms=stat(ranks,'miss_h2d_then_d2d_overlap_next_h2d'),p90_ms=p90(ranks,'miss_h2d_then_d2d_overlap_next_h2d')),
  miss_serial=dict(median_ms=stat(ranks,'miss_h2d_h2d_d2d_serial'),p90_ms=p90(ranks,'miss_h2d_h2d_d2d_serial')),
  activation=activation,
  expert_compute=compute,
 )

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 envs={name:env(a.root,name) for name in ('env1','env2')}
 for e in envs.values():
  e['miss_overlap_saving_vs_serial']=1-e['miss_overlap']['median_ms']/e['miss_serial']['median_ms']
 result=dict(status='PASS',environments=envs,
  interpretation='4-GPU model-free calibration; pair costs can calibrate the R8 CPU phase model but do not reproduce R8 collective contention')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
