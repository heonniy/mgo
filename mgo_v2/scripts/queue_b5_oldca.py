"""Owner-added Old CA, strictly after completion of active B5 diagnostics."""
import os,json,time
from pathlib import Path
from prepare_critical_microbench import ROOT,PACKET,P
from refactor_fingerprint import sha
from run_refactor_measure import publish

def main():
 os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005'
 deadline=time.monotonic()+3600
 while not (PACKET/'B5_AUDIT.json').exists():
  state=ROOT/'b5/C30/diagnostic_status.json'
  if state.exists() and json.loads(state.read_text())['status']=='FAIL':raise RuntimeError('B5 diagnostic failed; preserve gate before Old CA')
  if time.monotonic()>deadline:raise TimeoutError('B5 diagnostics/report did not finish')
  time.sleep(5)
 assert json.loads((PACKET/'B5_AUDIT.json').read_text())['status']=='PASS'
 import run_b5_oldca
 run_b5_oldca.main()
 base=ROOT/'b5_oldca/C30/B5_OLD_CA_C30_CLEAN_B128_H64'
 state=json.loads((base/'status.json').read_text());assert state['status']=='PASS'
 assert all(sha(P/k)==v for k,v in state['common_stack']['code'].items())
 result=json.loads((base/'result.json').read_text());old=result['gates']['OLD_CA_H1b']
 prior=json.loads((ROOT/'b5/C30/B5_C30_CLEAN_B128_H64/result.json').read_text())
 rows=[]
 for r in range(4):
  x=json.loads((base/f'OLD_CA_H1b_correctness_rank{r}.json').read_text());assert x['status']=='PASS' and x['post_expert_barriers']==3072;rows.append(dict(rank=r,**x))
 receipt=PACKET/'B5_OLD_CA_RESULTS.json'
 receipt.write_text(json.dumps(dict(status='PASS',primary_timing_stable=not result['unstable'],result=result,correctness=rows,measured_code_unchanged=True,comparison_limitation='Sequential addition after BR/FCA, not an interleaved paired comparison; session drift remains possible.'),indent=2)+'\n')
 lines=['# B5 Old CA addition','','Same C30/local B128/R4 GPUs 0,1,4,5/frozen decode64/BF16/H1b/V3/P2/T2/post-expert barrier condition. No new trace.','','Canonical OLD_CA tokens, physical copies/bytes, controller and wire traffic passed; 3072 barriers per rank. No scientific retry.','','| Policy | TPOT (s) | E2E (s) | Repeats | TPOT range (s) |','|---|---:|---:|---:|---|']
 for policy,g,n in [('BR',prior['gates']['BR_H1b'],len(prior['samples']['BR_H1b'])),('FCA',prior['gates']['FCA_H1b'],len(prior['samples']['FCA_H1b'])),('OLD_CA',old,len(result['samples']['OLD_CA_H1b']))]:
  lines.append(f"| {policy} | {g['estimate']['TPOT']:.6f} | {g['estimate']['E2E_wall']:.6f} | {n} | {g['range']['TPOT']} |")
 lines+=['',f"Old CA timing unstable: {old['unstable']}. All valid repeats retained; two-repeat estimate is mean/median, three-repeat estimate is median.",'','Old CA was added after the BR/FCA packet, rather than interleaved with it. Differences are descriptive comparisons and can include session drift. BR/FCA were not rerun. Old CA extension adds clean physical timing only; BR/FCA have the separate first8 barrier diagnostics.','']
 md=PACKET/'B5_OLD_CA_RESULTS.md';md.write_text('\n'.join(lines))
 publish('results: complete owner-added B5 Old CA timing',[receipt,md])
if __name__=='__main__':main()
