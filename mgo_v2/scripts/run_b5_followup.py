"""Continue the one authorized B5 attempt; never retry a failed scientific gate."""
import json,time
from prepare_critical_microbench import ROOT,PACKET
from run_refactor_measure import publish

def main():
 import os
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 base=ROOT/'b5/C30';path=base/'B5_C30_CLEAN_B128_H64/status.json'
 deadline=time.monotonic()+3700
 while True:
  state=json.loads(path.read_text())
  if state['status']=='FAIL':raise RuntimeError('B5 clean gate failed; no diagnostic or retry')
  restore=base/'resident_models.json'
  if state['status']=='PASS' and restore.exists() and json.loads(restore.read_text())['unix']>=state['finished_unix']:break
  if time.monotonic()>deadline:raise TimeoutError('B5 clean attempt did not finish')
  time.sleep(5)
 import run_b5_diagnostic
 run_b5_diagnostic.main()
 import report_b5
 report_b5.main()
 publish('results: complete B5 post-expert barrier isolation',list(PACKET.glob('B5_*.json'))+[PACKET/'B5_RESULTS.md'])
if __name__=='__main__':main()
