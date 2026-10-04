"""Remeasure BR/CA with normalized pinned H2D and three communication stacks."""
import argparse,json
from pathlib import Path
from run_env_offload_cell import ROOT,discover_validated_plan,run
from validate_env_offload_plan import validate

def next_free_plan_repeat(cell,policy):
 repeat=0
 while (ROOT/f'{cell}_{policy}_env1_PLAN_{repeat}').exists():repeat+=1
 return repeat

def ensure_plan(cell,policy):
 plan=discover_validated_plan(cell,policy)
 if plan is not None:return plan
 repeat=next_free_plan_repeat(cell,policy)
 plan=run(cell,policy,'PLAN','env1',repeat,comm_mode='current',h2d_mode='pageable')
 validate(plan)
 assert json.loads((plan/'schedule_validation.json').read_text())['status']=='PASS'
 return plan

def main(a):
 packet=Path(__file__).resolve().parents[1]/'experiments'/'transport_stack_remeasure_20261004'
 manifest=dict(cell=a.cell,environment=a.environment,repeats=a.repeats,h2d_mode='pinned',h2d_stages=a.h2d_stages,policies={},transports=a.transports)
 for policy in a.policies:
  plan=ensure_plan(a.cell,policy);manifest['policies'][policy]=str(plan)
  for transport in a.transports:
   run(a.cell,policy,'COMPILE',a.environment,0,transport,'pinned',plan,a.h2d_stages)
   for repeat in range(a.repeats):run(a.cell,policy,'MEASURE',a.environment,repeat,transport,'pinned',plan,a.h2d_stages)
   run(a.cell,policy,'COUNTERS',a.environment,0,transport,'pinned',plan,a.h2d_stages)
 packet.mkdir(exist_ok=True);(packet/'last_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',choices=['P','R','E'],default='R');p.add_argument('--environment',choices=['env1','env2'],default='env1');p.add_argument('--policies',nargs='+',choices=['BR','CA','CA-rep'],default=['BR','CA']);p.add_argument('--transports',nargs='+',choices=['current','coslot','coslot-active'],default=['current','coslot','coslot-active']);p.add_argument('--repeats',type=int,default=3);p.add_argument('--h2d-stages',type=int,default=2);main(p.parse_args())
