"""Bounded B4 full64 gate, preserving all results before timing authorization."""
import json,os,time
import run_refactor_measure as m
from prepare_critical_microbench import ROOT,PACKET,OLD
from batch_comm_common import stop_idle_load,start_idle_load

def main():
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 m.PACKET=PACKET;m.ROOT=ROOT/'b4'/'C30';m.ROOT.mkdir(parents=True,exist_ok=True);m.h.ROOT=m.ROOT
 inputs=m.ROOT/'inputs_B128_H64'
 if not inputs.exists():inputs.symlink_to(OLD/'C30'/'inputs_B128_H64')
 stop_idle_load()
 try:
  deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in m.h.snapshot()['gpus'] if g['gpu'] in [0,1,4,5]) and time.monotonic()<deadline:time.sleep(5)
  cases=[dict(label=p,policy=p,executor='H2',P=2,trigger='T2',horizon=64,overlap=True,partial_precision='bf16',runtime_arm='V3_OPT_PF_OVERLAP',staging_backend='torch',unique_combine=True,async_metadata_inputs=True,isolated_cpu_threads=True,fixed_staging_team=True) for p in ('BR','FCA')]
  state=m.run('B4_C30_CORRECTNESS_RETRY1',dict(environment='env1',world=4,gpus=[0,1,4,5],batch=128,horizon=64,paired=True,b3_executor_study=True,b4_correctness=True,cases=cases))
  m.h.write(PACKET/'B4_CORRECTNESS.json',state['result'])
  m.publish('B4: full64 correctness gate',[PACKET/'B4_CORRECTNESS.json'])
 finally:m.h.write(m.ROOT/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':main()
