# Fetch/communication Pareto: blocked at the transport prerequisite

**Status: BLOCKED_TRANSPORT. The requested Pareto study is not complete.** The four burn workers on GPUs 0,1,4,5 were stopped. T0 smoke and three T0 calibration cells passed. The T1 smoke failed before model loading; no routing capture, replica sweep, primary generation or profile was run. This is not a negative result about replication.

## Observed transport failure

With exactly the requested controls (`NCCL_P2P_DISABLE=1`, SHM and P2P_LEVEL overrides absent), NCCL 2.28.9 selected `NET/IB/.../GDRDMA/Shared`. The first `all_to_all_single` raised `ncclRemoteError` with `IBV_WC_RETRY_EXC_ERR(12)` and `vendor_err=129`. T0 selected `P2P/CUMEM` and completed the same transfers.

T1 logs contain no `via P2P/` or `via SHM/` channel. The path must not be called PCIe-only or verified host-staged SHM: `GDRDMA` is the recorded transport. Although SHM was left enabled, it was not selected. The logs describe T1 as `nNodes 4 localRanks 1`, versus T0 `nNodes 1 localRanks 4`; the reason for that topology decision remains unresolved. This was an explicit network completion error, not merely a slow model or an OOM.

All four workers shared the same IPC/mount/UTS namespaces, and `/dev/shm` had approximately 945 GiB available. No `NCCL_*` overrides were inherited and no `/etc/nccl.conf` or user `.nccl.conf` was present. These observations do not establish the underlying network/topology cause.

## Completed T0 calibration

Per rank: one 9 MiB pinned-host H2D copy; a BF16 decode communication proxy with hidden size 2048, eight dispatch rows and sixteen exact-expert return rows per peer. Peer bytes exclude self traffic and metadata. Ten warmups and thirty timed iterations per case. The table takes the maximum rank CUDA interval on each iteration, then reports median/p90. Concurrent copies and collectives are issued on separate streams; interval timing alone does not prove the amount of overlap. This proxy is not an end-to-end MoE latency or a peak fabric-bandwidth benchmark.

| Case | H2D median / p90 ms | Peer median / p90 ms | H2D GB/s | Peer GB/s |
|---|---:|---:|---:|---:|
| h2d | 0.2005 / 0.2273 | — / — | 47.0767 | — |
| peer | — / — | 0.3365 / 0.3723 | — | 0.8765 |
| concurrent | 0.1926 / 0.1961 | 0.2296 / 0.3018 | 48.9928 | 1.2844 |

Concurrent / standalone median ratios: H2D 0.961x; peer 0.682x. T1 calibration is missing because its mandatory smoke failed; no cross-transport cost conclusion is supported.

## Scope, evidence and next step

The prerequisite failure stopped subsequent GPU work. The selected GPUs were released; GPUs 2,3,6,7 and their existing workloads were untouched. No model was loaded, no OOM occurred, and no policy/runtime defaults changed. The empty Pareto/physical CSVs intentionally contain headers only.

Small INFO logs and worker exceptions are under `transport_logs/`; raw per-rank calibration samples remain under `/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002` with hashes in `raw_receipts.json`. The launcher actually used is preserved there as `initial_launcher.py`. The checked-in launcher adds timeout/memory bounds and requires a fresh output directory.

Resolve and re-verify the T1 transport first. Do not silently disable InfiniBand, force sockets/SHM, or relabel the observed path to manufacture the planned comparison. Any revised transport condition must be recorded explicitly before the model matrix is resumed. No replica implementation or final weighted policy was added.

Reproduction after the transport issue is addressed: run `PYTHONPATH=mgo_v2 /home/hwlee/sub-moe/phase01/.venv/bin/python mgo_v2/scripts/run_fetch_comm_calibration.py --root <fresh-output-directory>`. INFO logging is confined to the smoke; calibration subprocesses unset INFO logging.
