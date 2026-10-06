"""Isolated grouped service calibration, never fitted to policy TPOT."""
import argparse,json,csv,hashlib,statistics
from pathlib import Path
import numpy as np
import torch
from mgo_v2.grouped_expert import GroupedExpertExecutor

@torch.inference_mode()
def main(a):
 torch.set_num_threads(2);torch.manual_seed(42)
 vectors=[]
 for path in sorted(a.source.glob('*_waves_rank*.json')):
  for event in json.loads(path.read_text()):
   for wave in event['waves']:vectors.append(tuple(wave['rows']))
 assert vectors
 # Structural coverage only: group count, total rows, and skew. Deterministic
 # percentiles, without inspecting measured policy performance.
 samples={}
 for k in sorted(set(len(v) for v in vectors)):
  candidates=sorted(set(v for v in vectors if len(v)==k),key=lambda v:(sum(v),max(v),v))
  if k not in (1,2,4,8,16,32) and k%8:continue
  for q in (0,.5,1):
   v=candidates[round(q*(len(candidates)-1))];samples[v]='real_structural'
 # Include balanced/unbalanced matched-total controls at representative totals.
 totals=sorted(set(int(np.quantile([sum(v) for v in vectors],q)) for q in (.1,.5,.9)))
 max_k=max(map(len,vectors))
 for k in (1,2,4,8,16,32):
  if k>max_k:continue
  for n in totals:
   if n<k:continue
   samples[tuple(n//k+(i<n%k) for i in range(k))]='balanced'
   samples[tuple([n-k+1]+[1]*(k-1))]='skewed'
 kmax=max(map(len,samples));w=torch.randn((kmax,4718592),device='cuda',dtype=torch.bfloat16)*.02
 ex=GroupedExpertExecutor(w,None,max_rows=4096);ex.x.normal_();rows=[]
 for vector,kind in sorted(samples.items(),key=lambda x:(len(x[0]),sum(x[0]),x[0])):
  off=0;metadata=[]
  for slot,n in enumerate(vector):metadata.append([slot,off,n,off]);off+=n
  meta=torch.tensor(metadata,device='cuda',dtype=torch.int32)
  for _ in range(3):ex.math(meta,vector,off)
  torch.cuda.synchronize();timings=[]
  for _ in range(20):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   start.record();ex.math(meta,vector,off);end.record();end.synchronize();timings.append(start.elapsed_time(end))
  rows.append(dict(kind=kind,rows=list(vector),experts=len(vector),total_rows=off,max_rows=max(vector),row_tiles=sum((n+31)//32 for n in vector),gpu_ms=statistics.median(timings),samples_ms=timings))
 # Nonnegative additive features fitted only to isolated micro-calibration.
 from scipy.optimize import nnls
 features=lambda r:[1,r['experts'],r['total_rows'],r['row_tiles'],r['experts']*((r['max_rows']+31)//32)]
 x=np.array([features(r) for r in rows],float);y=np.array([r['gpu_ms'] for r in rows])
 coefficients,_=nnls(x,y)
 result=dict(status='PASS',scope='three grouped math kernels only; excluding gather, metadata, routing weight and readiness',fit_target='isolated CUDA-event grouped service, never TPOT',feature_order=['intercept','experts','total_rows','row_tiles','grid_row_tiles'],coefficients=coefficients.tolist(),rows=rows,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.source.glob('*_waves_rank*.json'))},workspace_bytes=ex.workspace_bytes)
 a.output.write_text(json.dumps(result,indent=2)+'\n')
 with a.output.with_suffix('.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
