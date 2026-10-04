#!/usr/bin/env python3
"""Exclusive GPU calibration, checkpoint, then calibrated CPU replay and GO."""
import argparse,json,os,subprocess,time
from pathlib import Path
from batch_comm_common import stop_idle_load
import run_timing_stability as harness
P=Path(__file__).resolve().parents[1]
ROOT=Path('/home/hwlee/mgo-results/r8_real_replica_batch_scaling_20261004')
PACKET=P/'experiments/r8_real_replica_batch_scaling_20261004'
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'
BRANCH='codex/r8-b128-b256-real-replica-20261004'
harness.ROOT=ROOT

def write(state):(PACKET/'execution_status.json').write_text(json.dumps(state,indent=2)+'\n')
def publish(message):
 subprocess.run(['git','add',str(PACKET)],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  rc=subprocess.run(['git','push','origin','HEAD:'+BRANCH],cwd=P.parent).returncode
  if rc:(ROOT/'publication_pending.json').write_text(json.dumps(dict(status='LOCAL_CHECKPOINT_SAVED_PUSH_PENDING')))

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,4,5');a=p.parse_args()
 assert a.gpus=='0,1,4,5','owner-selected GPUs'
 ROOT.mkdir(parents=True,exist_ok=True)
 state=dict(status='RUNNING',stage='MICROBENCH',gpus=[0,1,4,5],owner_commit='dbe17f2',started_unix=time.time());write(state)
 restored=False
 try:
  stop_idle_load()
  from run_replica_phase_microbench import snapshot
  deadline=time.monotonic()+180
  while any(x['temp_c']>=65 for x in snapshot([0,1,4,5]).values()) and time.monotonic()<deadline:time.sleep(5)
  ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
  target={line.split(',')[1].strip() for line in ids.splitlines() if int(line.split(',')[0]) in [0,1,4,5]}
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  assert not any(line.split(',')[0].strip() in target for line in apps.splitlines()),'selected GPUs occupied by foreign process'
  assert any(line.startswith('MemAvailable:') and int(line.split()[1])*1024>=256*2**30 for line in Path('/proc/meminfo').read_text().splitlines())
  subprocess.run([PYTHON,'-u',str(P/'scripts/run_replica_phase_microbench.py'),'--gpus',a.gpus],cwd=P,check=True)
  cal=json.loads((PACKET/'microbench_calibration.json').read_text());assert cal['status']=='PASS' and set(cal['environments'])=={'env1','env2'}
  state.update(stage='CALIBRATION_PASS');write(state);publish('results: validate Env1 Env2 calibration before replica CPU replay')
  state['resident_models']=harness.restore();restored=True
  state.update(stage='CPU_REPLAY');write(state)
  subprocess.run([PYTHON,'-u',str(P/'scripts/run_r8_phase_aware_policy_cpu.py')],cwd=P,check=True)
  decision=json.loads((PACKET/'GO_DECISION.json').read_text())
  state.update(status='COMPLETE',stage='GO_DECISION_READY',finished_unix=time.time(),decision_path=str(PACKET/'GO_DECISION.json'));write(state);publish('results: finish calibrated replica GO packet')
 except BaseException as exc:
  state.update(status='FAILED_OR_STOPPED',error=repr(exc),finished_unix=time.time());write(state);publish('results: preserve calibrated replica packet failure');raise
 finally:
  if not restored:state['resident_models']=harness.restore();write(state)

if __name__=='__main__':main()
