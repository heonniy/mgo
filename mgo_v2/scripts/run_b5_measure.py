"""Owner-authorized H1b-only C30 post-expert barrier isolation."""
import json,os,time
import run_refactor_measure as m
from prepare_critical_microbench import ROOT,PACKET,OLD
from batch_comm_common import stop_idle_load,start_idle_load

def main():
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 m.PACKET=PACKET;m.ROOT=ROOT/'b5/C30';m.h.ROOT=m.ROOT
 m.ROOT.mkdir(parents=True,exist_ok=True)
 inputs=m.ROOT/'inputs_B128_H64'
 if not inputs.exists():inputs.symlink_to(OLD/'C30/inputs_B128_H64')
 cases=json.loads((PACKET/'B5_C30_H1B_POST_EXPERT_BARRIER_CASES.json').read_text())
 assert [c['policy'] for c in cases]==['BR','FCA']
 assert all(c['executor']=='H1b' and c['post_expert_barrier'] is True and c['horizon']==64 for c in cases)
 stop_idle_load()
 try:
  deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in m.h.snapshot()['gpus'] if g['gpu'] in [0,1,4,5]) and time.monotonic()<deadline:time.sleep(5)
  return m.run('B5_C30_CLEAN',dict(environment='env1',world=4,gpus=[0,1,4,5],batch=128,horizon=64,paired=True,b3_executor_study=True,b5_measure=True,cases=cases))
 finally:m.h.write(m.ROOT/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':main()
