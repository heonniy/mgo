"""One bounded stage, burn paused for measurements and resumed in finally."""
import argparse,json
from batch_comm_common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['capture4','capture16','capture32','timing']);a=p.parse_args()
 state=json.loads((PACKET/'batch_comm_progress.json').read_text());assert a.stage not in state['completed']
 state.update(status='RUNNING',active_stage=a.stage)
 sources=[PACKAGE/'examples'/n for n in ('batch_comm_capture.py','trace_comm_worker.py','batch_comm_latency.py','ipc_rebase_smoke.py')]+[Path(__file__),PACKAGE/'scripts/batch_comm_common.py',PACKAGE/'scripts/trace_comm_input.py',PACKET/'batch_comm_prompt_provenance.json']
 state.setdefault('source_sha256',{str(p):sha(p) for p in sources});assert all(sha(p)==h for p,h in state['source_sha256'].items())
 write(PACKET/'batch_comm_progress.json',state)
 try:
  stop_burn()
  if a.stage.startswith('capture'):
   batch=int(a.stage[7:]);run(a.stage,'batch_comm_capture.py',['--batch',batch,'--prompts',PACKET/'batch_comm_prompt_provenance.json'],model=True)
  else:
   for mode in ('T0','R3'):run('smoke_'+mode,'ipc_rebase_smoke.py',['--mode',mode],mode=mode,smoke=True)
   for index,mode in ORDER:run(f'latency_p{index}_{mode}','batch_comm_latency.py',['--mode',mode],mode=mode)
   for batch in BATCHES:
    counts=PACKET/'trace_comm_counts.json' if batch==8 else ROOT/f'counts_B{batch}.json'
    for index,mode in ORDER:
     run(f'B{batch}_p{index}_{mode}','trace_comm_worker.py',['--counts',counts,'--mode',mode,'--pass-index',index,'--ipc-rebase'],mode=mode)
  state['completed'].append(a.stage);state.update(status='STAGE_PASS',active_stage=None)
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['burn_processes']=start_burn();write(PACKET/'batch_comm_progress.json',state)
if __name__=='__main__':main()
