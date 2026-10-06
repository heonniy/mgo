"""Two separate first8 barrier mechanism captures after clean full64 timing."""
import json,os,subprocess,time
from pathlib import Path
import run_timing_stability as h
from batch_comm_common import start_idle_load
from run_b2_diagnostic import NSYS,P,ROOT,OLD,PACKET

def main():
 gate=json.loads((ROOT/'b5/C30/B5_C30_CLEAN_B128_H64/status.json').read_text());assert gate['status']=='PASS'
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005';os.environ['MGO_NSYS_BINARY']=NSYS
 base=ROOT/'b5/C30';inputs=base/'inputs_B128_H8'
 if not inputs.exists():inputs.symlink_to(ROOT/'b2/C30/inputs_B128_H8')
 state=dict(status='RUNNING',completed=[],started_unix=time.time(),gpus=[0,1,4,5])
 try:
  for mode,policy in [('H1b','BR'),('H1b','FCA')]:
   stage=f'B5_C30_{mode}';label=f'{stage}_V3_OPT_PF_OVERLAP_{policy}_B128_H8';out=base/label
   if (out/'status.json').exists():assert json.loads((out/'status.json').read_text())['status']=='PASS'
   else:
    case=json.loads((OLD/'C30'/f'profile_case_{policy}.json').read_text())
    case.update(horizon=8,b2_instrumentation=True,b2_allow_other_gpu_jobs=True,b5_diagnostic=True,post_expert_barrier=True,b3_executor='H1b',b3_cache='C30')
    path=base/f'case_{mode}_{policy}.json';h.write(path,case)
    state['active']=f'C30/{mode}/{policy}';h.write(base/'diagnostic_status.json',state)
    cmd=[h.PYTHON,'-u',str(P/'scripts/run_refactor_profile.py'),'--environment','env1','--root',str(base),'--gpus','0','1','4','5','--stage',stage,'--horizon','8','--arm','V3_OPT_PF_OVERLAP','--batch','128','--policy',policy,'--case-override',str(path),'--staging-backend','torch','--unique-combine','--async-metadata-inputs','--fixed-staging-team','--cuda-flush-ms','600000','--defer-idle-restore']
    with (base/f'{label}_driver.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   state['completed'].append(str(out));h.write(base/'diagnostic_status.json',state)
  state['status']='DIAGNOSTICS_COMPLETE'
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();h.write(base/'diagnostic_status.json',state);h.write(PACKET/'B5_DIAGNOSTIC_EXECUTION.json',state)
  h.write(base/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':main()
