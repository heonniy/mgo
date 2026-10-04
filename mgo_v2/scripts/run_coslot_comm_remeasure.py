"""Remeasure BR/CA with pinned H2D across cache-pressure regimes."""
import argparse,json
from pathlib import Path
from run_env_offload_cell import ROOT,discover_validated_plan,run
from validate_env_offload_plan import validate

def cache_tag(cache_ratio):
 return f'c{int(round(float(cache_ratio)*100)):02d}'

def sub_tag(substitution):
 return 's1' if substitution else 's0'

def next_free_plan_repeat(cell,policy,cache_ratio,substitution):
 tag=cache_tag(cache_ratio);stag=sub_tag(substitution);repeat=0
 while (ROOT/f'{cell}_{policy}_env1_PLAN_{tag}_{stag}_{repeat}').exists():repeat+=1
 return repeat

def ensure_plan(cell,policy,cache_ratio,substitution):
 plan=discover_validated_plan(cell,policy,cache_ratio,substitution)
 if plan is not None:return plan
 repeat=next_free_plan_repeat(cell,policy,cache_ratio,substitution)
 plan=run(
  cell,policy,'PLAN','env1',repeat,
  comm_mode='current',h2d_mode='pageable',
  cache_ratio=cache_ratio,substitution=substitution,
 )
 validate(plan)
 proof=json.loads((plan/'schedule_validation.json').read_text());assert proof['status']=='PASS'
 rank0=json.loads((plan/'rank0.json').read_text())
 assert abs(float(rank0['cache_ratio'])-float(cache_ratio))<1e-9
 assert bool(rank0['substitution'])==bool(substitution)
 return plan

def main(a):
 packet=Path(__file__).resolve().parents[1]/'experiments'/'transport_stack_remeasure_20261004'
 substitution=(a.substitution=='on')
 manifest=dict(
  cell=a.cell,environment=a.environment,repeats=a.repeats,
  h2d_mode='pinned',h2d_stages=a.h2d_stages,
  substitution=substitution,cache_ratios=a.cache_ratios,runs={},
 )
 for ratio in a.cache_ratios:
  tag=cache_tag(ratio);manifest['runs'][tag]=dict(cache_ratio=ratio,policies={})
  for policy in a.policies:
   plan=ensure_plan(a.cell,policy,ratio,substitution)
   manifest['runs'][tag]['policies'][policy]=dict(plan=str(plan),transports=a.transports)
   for transport in a.transports:
    run(a.cell,policy,'COMPILE',a.environment,0,transport,'pinned',plan,a.h2d_stages,ratio,substitution)
    for repeat in range(a.repeats):
     run(a.cell,policy,'MEASURE',a.environment,repeat,transport,'pinned',plan,a.h2d_stages,ratio,substitution)
    run(a.cell,policy,'COUNTERS',a.environment,0,transport,'pinned',plan,a.h2d_stages,ratio,substitution)
 packet.mkdir(exist_ok=True)
 (packet/'last_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--cell',choices=['P','R','E'],default='R')
 p.add_argument('--environment',choices=['env1','env2'],default='env1')
 p.add_argument('--policies',nargs='+',choices=['BR','CA','CA-rep'],default=['BR','CA'])
 p.add_argument('--transports',nargs='+',choices=['current','coslot','coslot-active'],default=['current','coslot','coslot-active'])
 p.add_argument('--cache-ratios',nargs='+',type=float,default=[0.60])
 p.add_argument('--substitution',choices=['off','on'],default='off')
 p.add_argument('--repeats',type=int,default=3)
 p.add_argument('--h2d-stages',type=int,default=2)
 a=p.parse_args();assert all(0.0<x<=1.0 for x in a.cache_ratios);main(a)
