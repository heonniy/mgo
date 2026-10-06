"""C30 H0/H1 diagnostics; full64 H1 preparation, fixed8 diagnostic prefix."""
import json, os, subprocess, time
from pathlib import Path
import run_timing_stability as h
from batch_comm_common import start_idle_load
from run_b2_diagnostic import NSYS, P, ROOT, OLD, PACKET
B3=ROOT/'b3'
def main(wrapper=False):
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005';os.environ['MGO_NSYS_BINARY']=NSYS
 base=B3/'C30';base.mkdir(parents=True,exist_ok=True)
 inputs=base/'inputs_B128_H8'
 if not inputs.exists():inputs.symlink_to(ROOT/'b2/C30/inputs_B128_H8')
 if wrapper:
  old=json.loads((B3/'status.json').read_text());assert old['status']=='DIAGNOSTICS_COMPLETE'
  host=json.loads((B3/'H1_HOST_ONLY_PRELIMINARY.json').read_text())
  assert all(x['reductions']['expert_compiled_kernel']>=.70 and x['reductions']['expert_loop']<.40 for x in host.values())
  initial=[p for p in old['completed'] if json.loads((Path(p)/'case.json').read_text())['b3_executor']=='H0']
 else:initial=[]
 statuspath=B3/('h1b_status.json' if wrapper else 'status.json')
 state=dict(status='RUNNING',completed=initial,started_unix=time.time(),gpus=[0,1,4,5])
 try:
  # H1 first resolves feasibility/correctness before spending new H0 captures.
  for mode,policy in ([('H1b','BR'),('H1b','FCA')] if wrapper else [('H1','BR'),('H0','FCA'),('H1','FCA'),('H0','BR')]):
   stage=f'B3_C30_NODE1_{mode}';label=f'{stage}_V3_OPT_PF_OVERLAP_{policy}_B128_H8';out=base/label
   if (out/'status.json').exists():
    assert json.loads((out/'status.json').read_text())['status']=='PASS','preserve failed attempts'
   else:
    case=json.loads((OLD/'C30'/f'profile_case_{policy}.json').read_text())
    case.update(horizon=8,b2_instrumentation=True,b2_allow_other_gpu_jobs=True,b3_executor=mode,b3_cache='C30')
    casepath=base/f'case_{mode}_{policy}.json';h.write(casepath,case)
    state['active']=f'C30/{mode}/{policy}';h.write(statuspath,state)
    cmd=[h.PYTHON,'-u',str(P/'scripts/run_refactor_profile.py'),'--environment','env1','--root',str(base),'--gpus','0','1','4','5','--stage',stage,'--horizon','8','--arm','V3_OPT_PF_OVERLAP','--batch','128','--policy',policy,'--case-override',str(casepath),'--staging-backend','torch','--unique-combine','--async-metadata-inputs','--fixed-staging-team','--cuda-flush-ms','600000','--defer-idle-restore']
    with (base/f'{label}_driver.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
   state['completed'].append(str(out));h.write(statuspath,state)
  state['status']='DIAGNOSTICS_COMPLETE'
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();h.write(statuspath,state);h.write(PACKET/('B3_H1b_EXECUTION.json' if wrapper else 'B3_EXECUTION.json'),state)
  h.write(B3/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':
 import sys
 main(wrapper='--wrapper' in sys.argv)
