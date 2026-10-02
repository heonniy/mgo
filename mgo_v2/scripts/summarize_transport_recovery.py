#!/usr/bin/env python3
"""Checkpoint each bounded smoke and the first successful path's calibration."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from run_transport_recovery import OUT,ROOT,PACKAGE,read,write,sha

def main():
 result=read(OUT/'transport_recovery_result.json')
 baseline=next(r for r in csv.DictReader((OUT/'transport_calibration.csv').open()) if r['mode']=='T0' and r['case']=='peer')
 if result.get('calibration',{} ) and result['calibration']['status']=='PASS':
  condition=result['selected_condition'];raw=Path(result['calibration']['raw_root'])
  ranks=[read(raw/f'rank{r}.json') for r in range(4)];rows=[]
  for i,case in enumerate(('h2d','peer','concurrent')):
   samples=[r['records'][i]['samples'] for r in ranks];assert all(len(s)==30 for s in samples)
   row=dict(condition=condition,case=case,warmups=10,timed_iterations=30,ranks=4)
   for kind in ('h2d','peer'):
    data=np.max([[x[kind+'_ms'] for x in s] for s in samples],axis=0)
    for q,label in ((50,'median'),(90,'p90')):row[kind+'_'+label+'_ms']=float(np.percentile(data,q)) if data.max() else None
   rows.append(row)
  ratio=rows[1]['peer_median_ms']/float(baseline['peer_median_ms'])
  result['calibration'].update(summary=rows,T0_peer_median_ms=float(baseline['peer_median_ms']),peer_cost_ratio_vs_T0=ratio)
  result['status']=('FUNCTIONAL_NO_COST_INCREASE' if ratio<=1 else
                    'READY_FOR_STAGE_1' if ratio>=2 else 'FUNCTIONAL_BUT_BELOW_2X_REPORT_BEFORE_CONTINUING')
  result['calibration']['cost_increase_demonstrated']=ratio>1
  result['calibration']['preferred_2x_contrast_met']=ratio>=2
  chosen=next(a for a in result['attempts'] if a['condition']==condition)
  chosen.update(peer_median_ms=rows[1]['peer_median_ms'],h2d_median_ms=rows[0]['h2d_median_ms'])
  with (OUT/'transport_recovery_calibration.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
  # Small per-rank raw samples make this comparison reproducible from Git.
  dest=OUT/'transport_recovery_logs'/(condition+'_calibration');dest.mkdir(exist_ok=True)
  for f in sorted(raw.glob('*.json')):(dest/f.name).write_bytes(f.read_bytes())
  (dest/'run.log').write_bytes((raw/'run.log').read_bytes())
 write(OUT/'transport_recovery_result.json',result)
 with (OUT/'transport_recovery.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['condition','status','p2p_used','selected_transport','error','peer_median_ms','h2d_median_ms','notes'],extrasaction='ignore',lineterminator='\n');w.writeheader()
  for a in result['attempts']:w.writerow(dict(a,notes='Bounded transport recovery; model not loaded.'))
 report=['# Bounded transport recovery','',f'Current status: **{result["status"]}**. Plan: `92b2070`. Original failed T1 and T0 calibration remain recorded at `7c881f7`.','',
 'Only physical GPUs 0,1,4,5 are used. Every smoke uses the same two all-to-all BF16 payloads and validates every returned row on all four ranks. The worker removes the bootstrap default `NCCL_P2P_DISABLE=0` before NCCL initialization for R conditions, leaving that variable truly unset. SHM remains enabled. No model, replica policy or system IB changes were introduced.','',
 '| Condition | Result | Direct P2P | Actual transport | Error |','|---|---|---|---|---|']
 for a in result['attempts']:report.append(f'| {a["condition"]} | {a["status"]} | {a["p2p_used"]} | {a["selected_transport"]} | {a["error"] or "—"} |')
 report+=['','Transport strings come from INFO channel lines; successful acceptance additionally requires payload validation on every rank and no P2P/NVLS channel. `GDRDMA` and `Shared` in a NET/IB line do not establish SHM transport.','']
 for a in result['attempts']:
  report += [f'## {a["condition"]}','',f'Explicit environment: `{json.dumps(a["environment"],sort_keys=True)}`. Other inherited `NCCL_*` variables were cleared; INFO was enabled only for smoke.','',
             'Topology lines (opaque communicator addresses are retained in the raw logs):','']
  report += ['- `'+line+'`' for line in a['topology_lines']]
  report += ['']
 if result.get('calibration') and result['calibration']['status']=='PASS':
  c=result['calibration'];report += ['## Calibration of the first valid path','',
   'Three cases only, each with 10 warmups and 30 measured iterations. Values summarize per-iteration maximum-rank CUDA intervals; timing method and payloads match the original T0 calibration. INFO logging is off. These small messages measure software/transfer latency as well as device activity, not peak fabric bandwidth.','',
   '| Case | H2D median ms | Peer median ms | H2D p90 ms | Peer p90 ms |','|---|---:|---:|---:|---:|']
  for r in c['summary']:
   def val(k):return '—' if r[k] is None else f'{r[k]:.6f}'
   report.append(f'| {r["case"]} | {val("h2d_median_ms")} | {val("peer_median_ms")} | {val("h2d_p90_ms")} | {val("peer_p90_ms")} |')
  report += ['',f'Peer median / original T0 ({c["T0_peer_median_ms"]:.6f} ms): **{c["peer_cost_ratio_vs_T0"]:.3f}x**. The original T0 was measured earlier on the shared host; this is a small calibration comparison, not an end-to-end claim.','']
  if c['peer_cost_ratio_vs_T0']<=1:report+=['The first functional path does not demonstrate more expensive communication on this calibration, so the required Stage 1 cost-increase gate is not met. The preferred 2x contrast is also absent. Stage 1 has not started. The smaller measured median does not establish that SHM is intrinsically faster: the runs were separated in time on a shared host. Do not try later R conditions or retune messages merely to obtain a larger ratio.','']
  elif c['peer_cost_ratio_vs_T0']<2:report+=['The first functional path did not meet the preferred 2x contrast. Per the recovery plan, report this result before continuing; Stage 1 has not started. Do not try later R conditions merely to obtain a larger ratio.','']
  else:report+=['The transport and preferred cost-contrast gates pass. Commit this calibration before starting Stage 1.','']
 report+=['The original `artifact_hashes.json` is the historical Stage 0 snapshot; owner edits at `92b2070` and this recovery are separate. Current recovery artifacts/source hashes are in `transport_recovery_hashes.json`. Raw trial commands, environment, source hashes and memory samples are retained alongside each smoke.','']
 (OUT/'TRANSPORT_RECOVERY_RESULTS.md').write_text('\n'.join(report))
 paths=[OUT/n for n in ('transport_recovery.csv','transport_recovery_result.json','TRANSPORT_RECOVERY_RESULTS.md','TRANSPORT_RECOVERY.md',
                       'validation.json','README.md','RESULTS.md','PLAN.md','AGENT_TASK.md','matrix.json')]
 paths+=list((OUT/'transport_recovery_logs').rglob('*'))
 if (OUT/'transport_recovery_calibration.csv').exists():paths.append(OUT/'transport_recovery_calibration.csv')
 sources=[PACKAGE/'examples/fetch_comm_calibration.py',PACKAGE/'scripts/run_transport_recovery.py',Path(__file__)]
 write(OUT/'transport_recovery_hashes.json',dict(artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted(paths) if p.is_file()},sources={str(p.relative_to(PACKAGE)):sha(p) for p in sources}))
 print(result['status'])
if __name__=='__main__':main()
