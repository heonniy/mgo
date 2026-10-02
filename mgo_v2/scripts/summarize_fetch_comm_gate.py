#!/usr/bin/env python3
"""Publish the observed transport failure without inventing unrun results."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002')
OUT=PACKAGE/'experiments/fetch_comm_pareto_p2p_20261002'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 assert read(ROOT/'T0_transport/status.json')['status']=='PASS'
 assert read(ROOT/'T0_calibration/status.json')['status']=='PASS'
 assert read(ROOT/'T1_transport/status.json')['status']=='FAIL'
 logs=OUT/'transport_logs';logs.mkdir(exist_ok=True)
 checks={}
 for mode in ('T0','T1'):
  files=list((ROOT/f'{mode}_transport').glob('nccl-*.log'))
  assert len(files)==4
  lines='\n'.join(f.read_text() for f in files)
  check=dict(direct_p2p_channels=lines.count('via P2P/'),shm_channels=lines.count('via SHM/'),
             ib_gdr_channels=lines.count('via NET/IB/'),ib_retry_errors=lines.count('IBV_WC_RETRY_EXC_ERR'),
             nvls_channels=lines.count('via NVLS'),status=read(ROOT/f'{mode}_transport/status.json')['status'])
  checks[mode]=check
  for f in files:shutil.copyfile(f,logs/f'{mode}-{f.name}')
  shutil.copyfile(ROOT/f'{mode}_transport/run.log',logs/f'{mode}-worker.log')
 assert checks['T0']['direct_p2p_channels']>0
 assert checks['T1']['direct_p2p_channels']==0 and checks['T1']['shm_channels']==0
 assert checks['T1']['ib_gdr_channels']>0 and checks['T1']['ib_retry_errors']>0
 ranks=[read(ROOT/f'T0_calibration/rank{r}.json') for r in range(4)]
 rows=[]
 for i,case in enumerate(('h2d','peer','concurrent')):
  samples=[d['records'][i]['samples'] for d in ranks];assert all(len(s)==30 for s in samples)
  row=dict(mode='T0',case=case,status='PASS',warmups=10,timed_iterations=30,ranks=4,
           h2d_bytes_per_rank=9*1024**2 if case!='peer' else 0,
           peer_tx_bytes_per_rank=3*(8+16)*2048*2 if case!='h2d' else 0)
  for field in ('h2d','peer'):
   data=np.max([[s[field+'_ms'] for s in rs] for rs in samples],axis=0)
   for q,name in ((50,'median'),(90,'p90')):row[f'{field}_{name}_ms']=float(np.percentile(data,q)) if data.max() else None
   n=row['h2d_bytes_per_rank' if field=='h2d' else 'peer_tx_bytes_per_rank']
   row[f'{field}_effective_GBps']=n/row[f'{field}_median_ms']/1e6 if n else None
  rows.append(row)
 for field,solo in (('h2d',rows[0]),('peer',rows[1])):
  rows[2][field+'_concurrent_slowdown']=rows[2][field+'_median_ms']/solo[field+'_median_ms']
 for case in ('h2d','peer','concurrent'):rows.append(dict(mode='T1',case=case,status='NOT_RUN_TRANSPORT_FAILED'))
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/'transport_calibration.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 # Deliberate empty tables: no policy or physical measurements were made.
 (OUT/'pareto_points.csv').write_text('replica_budget,expert_h2d_bytes,peer_activation_bytes,nondominated\n')
 (OUT/'physical_runs.csv').write_text('point,transport,repeat,tpot_seconds,generation_seconds,status\n')
 receipts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(list(ROOT.glob('*/*'))+[ROOT/'initial_launcher.py']) if p.is_file()]
 (OUT/'raw_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
 validation=dict(status='BLOCKED_TRANSPORT',study_complete=False,plan_commit='f779eef70b08c8ac75c78872f08d3dcaacbbf8ed',
     physical_gpus=[0,1,4,5],burn_workers_stopped=True,model_generations=0,calibration_cells_completed=3,
     calibration_cells_planned=6,trace_capture=False,cpu_replica_sweep=False,primary_generations=0,profiles=0,
     transport=checks,blocker='T1 all_to_all_single selects NET/IB/GDRDMA/Shared and fails with IBV_WC_RETRY_EXC_ERR; SHM fallback is unverified.',
     outcome='Transport prerequisite blocked; no conclusion about the replica Pareto frontier.',
     source_sha256={str(p.relative_to(PACKAGE)):sha(p) for p in [PACKAGE/'examples/fetch_comm_calibration.py',PACKAGE/'scripts/run_fetch_comm_calibration.py',Path(__file__)]},
     run_provenance='The recorded worker ran under the saved temporary launcher; the checked-in launcher adds bounded timeout/memory guards for a deliberate rerun. No calibration or model measurements were rerun.',
     unmodified_transport_controls=['NCCL_SHM_DISABLE unset','NCCL_P2P_LEVEL unset','No NCCL_IB_DISABLE override','T0 NCCL_P2P_DISABLE unset','T1 NCCL_P2P_DISABLE=1'],
     resource_scope='Tiny calibration only: one 9 MiB pinned source/destination and 384 KiB send/receive buffers per rank, plus CUDA/NCCL context. Observed approximately 1.1 GiB GPU use per rank. No model loaded; selected GPUs returned to 0 MiB after failure. No OOM observed.')
 (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 report=['# Fetch/communication Pareto: blocked at the transport prerequisite','',
 '**Status: BLOCKED_TRANSPORT. The requested Pareto study is not complete.** The four burn workers on GPUs 0,1,4,5 were stopped. T0 smoke and three T0 calibration cells passed. The T1 smoke failed before model loading; no routing capture, replica sweep, primary generation or profile was run. This is not a negative result about replication.','',
 '## Observed transport failure','',
 'With exactly the requested controls (`NCCL_P2P_DISABLE=1`, SHM and P2P_LEVEL overrides absent), NCCL 2.28.9 selected `NET/IB/.../GDRDMA/Shared`. The first `all_to_all_single` raised `ncclRemoteError` with `IBV_WC_RETRY_EXC_ERR(12)` and `vendor_err=129`. T0 selected `P2P/CUMEM` and completed the same transfers.','',
 'T1 logs contain no `via P2P/` or `via SHM/` channel. The path must not be called PCIe-only or verified host-staged SHM: `GDRDMA` is the recorded transport. Although SHM was left enabled, it was not selected. The logs describe T1 as `nNodes 4 localRanks 1`, versus T0 `nNodes 1 localRanks 4`; the reason for that topology decision remains unresolved. This was an explicit network completion error, not merely a slow model or an OOM.','',
 'All four workers shared the same IPC/mount/UTS namespaces, and `/dev/shm` had approximately 945 GiB available. No `NCCL_*` overrides were inherited and no `/etc/nccl.conf` or user `.nccl.conf` was present. These observations do not establish the underlying network/topology cause.','',
 '## Completed T0 calibration','',
 'Per rank: one 9 MiB pinned-host H2D copy; a BF16 decode communication proxy with hidden size 2048, eight dispatch rows and sixteen exact-expert return rows per peer. Peer bytes exclude self traffic and metadata. Ten warmups and thirty timed iterations per case. The table takes the maximum rank CUDA interval on each iteration, then reports median/p90. Concurrent copies and collectives are issued on separate streams; interval timing alone does not prove the amount of overlap. This proxy is not an end-to-end MoE latency or a peak fabric-bandwidth benchmark.','',
 '| Case | H2D median / p90 ms | Peer median / p90 ms | H2D GB/s | Peer GB/s |',
 '|---|---:|---:|---:|---:|']
 def fmt(v):return '—' if v is None else f'{v:.4f}'
 for row in rows[:3]:
  report.append(f'| {row["case"]} | {fmt(row["h2d_median_ms"])} / {fmt(row["h2d_p90_ms"])} | {fmt(row["peer_median_ms"])} / {fmt(row["peer_p90_ms"])} | {fmt(row["h2d_effective_GBps"])} | {fmt(row["peer_effective_GBps"])} |')
 report+=['',f'Concurrent / standalone median ratios: H2D {rows[2]["h2d_concurrent_slowdown"]:.3f}x; peer {rows[2]["peer_concurrent_slowdown"]:.3f}x. T1 calibration is missing because its mandatory smoke failed; no cross-transport cost conclusion is supported.','',
 '## Scope, evidence and next step','',
 'The prerequisite failure stopped subsequent GPU work. The selected GPUs were released; GPUs 2,3,6,7 and their existing workloads were untouched. No model was loaded, no OOM occurred, and no policy/runtime defaults changed. The empty Pareto/physical CSVs intentionally contain headers only.','',
 'Small INFO logs and worker exceptions are under `transport_logs/`; raw per-rank calibration samples remain under `/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002` with hashes in `raw_receipts.json`. The launcher actually used is preserved there as `initial_launcher.py`. The checked-in launcher adds timeout/memory bounds and requires a fresh output directory.','',
 'Resolve and re-verify the T1 transport first. Do not silently disable InfiniBand, force sockets/SHM, or relabel the observed path to manufacture the planned comparison. Any revised transport condition must be recorded explicitly before the model matrix is resumed. No replica implementation or final weighted policy was added.','',
 'Reproduction after the transport issue is addressed: run `PYTHONPATH=mgo_v2 /home/hwlee/sub-moe/phase01/.venv/bin/python mgo_v2/scripts/run_fetch_comm_calibration.py --root <fresh-output-directory>`. INFO logging is confined to the smoke; calibration subprocesses unset INFO logging.']
 (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n')
 (OUT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'},indent=2)+'\n')
 print(json.dumps(dict(status=validation['status'],transport=checks,calibration=rows[:3])))
if __name__=='__main__':main()
