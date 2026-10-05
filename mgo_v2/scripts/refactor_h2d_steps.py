"""Rank-local per-step DMA overlap; never equates overlap with causal speedup."""
from collections import defaultdict
import statistics

def summarize_rank(row,horizon):
 assert row['status']=='PASS' and row['decode_events']==48*horizon
 steps=defaultdict(lambda:defaultdict(float));layers=defaultdict(set)
 for event in row['interval_decomposition']['per_event']:
  step=event['step'];assert 0<=step<horizon
  layers[step].add(event['layer'])
  for key,value in event['exclusive_ms'].items():steps[step][key]+=value
  a,b=event['window_ns'];steps[step]['window_ms']+=(b-a)/1e6
 assert set(steps)==set(range(horizon)) and all(v==set(range(48)) for v in layers.values())
 values=[]
 for step,bins in sorted(steps.items()):
  exposed=bins['H2D_only'];hidden=sum(bins[k] for k in ('H2D_COMM','H2D_COMPUTE','H2D_COMM_COMPUTE'))
  values.append(dict(step=step,total_H2D_ms=exposed+hidden,overlapped_H2D_ms=hidden,outside_COMM_COMPUTE_H2D_ms=exposed,observed_window_ms=bins['window_ms']))
 averages={key:statistics.mean(x[key] for x in values) for key in values[0] if key!='step'}
 assert abs(sum(x['total_H2D_ms'] for x in values)-row['H2D']['total_union_ms'])<1e-5
 return dict(per_step=values,mean_per_step=averages,cpu_staging_wall_ms_per_step=(row['cpu_nvtx']['moe.host_staging']['total_ms']/horizon if 'moe.host_staging' in row['cpu_nvtx'] else None),scope='Actual DMA time union on one rank; COMM includes metadata plus payload NCCL. Adjacent host MoE-event windows partition decode; windows are not causal ownership or full-model TPOT. Outside-COMM/COMPUTE DMA is exposed by this overlap definition, not proven critical-path latency. Do not sum ranks or infer TPOT savings from overlap alone.')
