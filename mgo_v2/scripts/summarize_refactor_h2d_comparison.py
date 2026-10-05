"""Per-step actual DMA comparison, with clean timing kept separate."""
import json,statistics,subprocess,hashlib
from pathlib import Path
from prepare_refactor_arms import ARMS,PACKET
from refactor_h2d_steps import summarize_rank
import run_timing_stability as h
ROOT=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/R4_H64')

def main():
 state=json.loads((ROOT/'H2D_COMPARISON_STATUS.json').read_text());assert state['status']=='CAPTURES_COMPLETE'
 timing=json.loads((PACKET/'STABLE_THREE_ARM_RESULTS.json').read_text());rows=[];sources={}
 for path in state['completed']:
  capture=Path(path);case=json.loads((capture/'case.json').read_text());analysis=capture/'interval_analysis'
  if not (analysis/'summary.json').exists():
   assert not analysis.exists(),'incomplete export requires an explicitly new output folder'
   with (capture/'export_driver.log').open('w') as log:subprocess.run([h.PYTHON,str(h.P/'scripts/export_refactor_profiles.py'),str(capture)],stdout=log,stderr=subprocess.STDOUT,check=True)
  ranks=[]
  for rank in range(4):
   f=analysis/f'rank{rank}_intervals.json';raw=f.read_bytes();sources[str(f)]=hashlib.sha256(raw).hexdigest()
   ranks.append(dict(rank=rank,**summarize_rank(json.loads(raw),64)))
  batch=int(capture.name.split('_B')[-1].split('_')[0]);arm=case['runtime_arm'];policy=case['policy']
  t=next(x for x in timing['rows'] if x['arm']==arm)['batches'][str(batch)]
  clean={metric:statistics.mean(p[policy][metric] for p in t['samples']) for metric in ('E2E_wall','TPOT')}
  distribution={k:dict(mean_across_ranks=statistics.mean(x['mean_per_step'][k] for x in ranks),min_rank=min(x['mean_per_step'][k] for x in ranks),max_rank=max(x['mean_per_step'][k] for x in ranks)) for k in ranks[0]['mean_per_step']}
  rows.append(dict(arm=arm,batch=batch,policy=policy,clean_timing=clean,profile_rank_mean_per_step_ms=distribution,ranks=ranks,capture=str(capture)))
 for b in (128,256):
  base=next(x for x in rows if (x['arm'],x['batch'],x['policy'])==(ARMS[0],b,'LA'))
  for x in rows:
   if x['batch']!=b or x['policy']!='LA':continue
   x['versus_no_prefetch']=dict(TPOT_reduction_ms=(base['clean_timing']['TPOT']-x['clean_timing']['TPOT'])*1000,TPOT_reduction_fraction=1-x['clean_timing']['TPOT']/base['clean_timing']['TPOT'],mean_rank_outside_COMM_COMPUTE_H2D_reduction_ms=base['profile_rank_mean_per_step_ms']['outside_COMM_COMPUTE_H2D_ms']['mean_across_ranks']-x['profile_rank_mean_per_step_ms']['outside_COMM_COMPUTE_H2D_ms']['mean_across_ranks'])
 report=dict(status='PASS',rows=rows,source_sha256=sources,units='Milliseconds per decode step for profile intervals; seconds for clean E2E and TPOT.',scope='H2D means actual 9-MiB expert DMA; CPU staging is separate. Rank means/min/max summarize four devices, never their sum. Overlap includes metadata/payload NCCL residency and expert kernels; NCCL may be waiting. Outside-overlap DMA is not a proven critical-path delay, and its change is not equated with clean TPOT savings. Profile observation windows are MoE event partitions, not full-model step latency. Every speed comparison uses separate uninstrumented repeats.')
 (PACKET/'H2D_PER_STEP_COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n')
 for x in rows:
  if x['policy']=='LA':print(x['arm'],x['batch'],x['clean_timing'],x['profile_rank_mean_per_step_ms'],x['versus_no_prefetch'])
if __name__=='__main__':main()
