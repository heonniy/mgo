"""Clean B3 timing, only after separately completed C30 diagnostic gate."""
import json,os,time
import run_refactor_measure as m
from prepare_critical_microbench import ROOT,PACKET,OLD
from batch_comm_common import stop_idle_load,start_idle_load

def main(cache='C30'):
 diagnosis=PACKET/'B3_H1b_DIAGNOSTIC_RESULTS.json'
 if not diagnosis.exists():diagnosis=PACKET/'B3_DIAGNOSTIC_RESULTS.json'
 gate=json.loads(diagnosis.read_text());candidate=gate.get('candidate','H1')
 assert gate['status']=='DIAGNOSTIC_PASS','correctness/host-repair diagnosis must pass first'
 assert cache in ('C30','C60')
 if cache=='C60':
  decision=json.loads((PACKET/'B3_REPAIR_DECISION.json').read_text())
  assert decision['status']=='REPAIR_PASS' and decision['c60_authorized']
  assert candidate==decision['candidate'],'no C60 retuning'

 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 m.PACKET=PACKET;m.ROOT=ROOT/'b3'/cache;m.h.ROOT=m.ROOT
 inputs=m.ROOT/'inputs_B128_H64'
 if not inputs.exists():inputs.symlink_to(OLD/cache/'inputs_B128_H64')
 stop_idle_load()
 try:
  deadline=time.monotonic()+180
  while any(g['temperature_c']>=65 for g in m.h.snapshot()['gpus'] if g['gpu'] in [0,1,4,5]) and time.monotonic()<deadline:time.sleep(5)
  cases=[dict(label=p,policy=p,executor=candidate,P=2,trigger='T2',horizon=64,overlap=True,partial_precision='bf16',runtime_arm='V3_OPT_PF_OVERLAP',staging_backend='torch',unique_combine=True,async_metadata_inputs=True,isolated_cpu_threads=True,fixed_staging_team=True) for p in ('BR','FCA')]
  return m.run('B3_'+cache+'_CLEAN',dict(environment='env1',world=4,gpus=[0,1,4,5],batch=128,horizon=64,paired=True,b3_executor_study=True,cases=cases))
 finally:m.h.write(m.ROOT/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--cache',choices=['C30','C60'],default='C30');main(p.parse_args().cache)
