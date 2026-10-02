# Bounded P2P/cuMem preflight diagnosis

**CUMEM_PATH_UNSTABLE**. Default T0 payload validation passed 0/3 fresh trials.

| Trial | cuMem override | Outcome | Selected path(s) | Wall s including cleanup |
|:---|:---|:---|:---|---:|
| D1_0 | unset | TIMEOUT | P2P/CUMEM | 91.961 |
| D1_1 | unset | TIMEOUT | P2P/CUMEM | 91.612 |
| D1_2 | unset | TIMEOUT | P2P/CUMEM | 91.763 |
| D2 | NCCL_CUMEM_ENABLE=0 | PASS | P2P/IPC | 10.720 |

Each trial sends exactly 32 KiB per peer, uses a fresh four-rank process group on GPUs 0,1,4,5, and has a 90-second active-runtime limit. Wall time above includes termination/cleanup after a timeout; timeouts are failures, not performance samples. PASS requires all four payload receipts; default trials additionally require P2P/CUMEM. The alternate path is reported as observed.

## Evidence and resource isolation

- Each trial has a prelaunch snapshot of all eight GPU process/utilization states, target free memory, host availability, ambient/launch NCCL environment, and CUDA/PyTorch/NCCL versions. Snapshots are committed as `cumem_<trial>_environment.json`.
- INFO logging is confined to these four diagnostics. Compact per-rank excerpts retain initialization/path evidence and final 50 lines; raw logs and receipts have hashes in the JSON.
- Timeout evidence records worker wait channels, the last process/GPU/host memory sample and the last 50 NCCL lines per rank before terminating only the trial process group.
- Peak trial process-tree RSS 4.669 GiB; minimum host available 1851.99 GiB; minimum target GPU free 78,439 MiB.
- No driver reset, server reboot, channel/protocol tuning, model generation or changes to workloads on GPUs 2,3,6,7. The cuMem-disabled trial is diagnostic only.

## Decision boundary

At least one default trial failed while the cuMem-disabled diagnostic passed. This supports cuMem-path-specific instability, without establishing the driver root cause. Stop after diagnosis: no E1 retry, no alternate experimental baseline, and no model work.

See [CSV](cumem_preflight_retry.csv), [JSON/provenance](cumem_preflight_retry.json), and [owner plan](CUMEM_PREFLIGHT_RETRY.md).
