# IPC baseline rebase and actual-trace communication gate

**AMBIGUOUS_GAP**. CPU Pareto points, F/K selection and frozen schedule metadata are unchanged (SHA256 verified); no CPU screen or model was run.

## I0 paired transport acceptance

| Mode | Acceptance | Observed channel paths |
|:---|:---|:---|
| T0-IPC (direct P2P / NVSwitch) | PASS | ['P2P/IPC'] |
| R3-SHM (P2P disabled / host-staged) | PASS | ['SHM/direct/direct'] |

Both modes set `NCCL_CUMEM_ENABLE=0`. T0-IPC clears other transport overrides; R3-SHM additionally sets P2P_LEVEL=LOC and IB_DISABLE=1. Accepted smokes require one node/four local ranks and all-rank payload validation. T0 permits only P2P/IPC; R3 permits only SHM with no P2P/NET. INFO is confined to I0.

I1 R3-SHM/T0-IPC ratios: **1.0529x** in pass 0 and **1.0836x** in pass 1; median **1.0682x**.

| Pass/order | Mode | Median cumulative max-rank ms | Three trace totals ms |
|:---|:---|---:|:---|
| 0 (T0 → R3) | T0-IPC (direct P2P / NVSwitch) | 41.3988 | 39.2597, 51.0556, 41.3988 |
| 0 (T0 → R3) | R3-SHM (P2P disabled / host-staged) | 43.5886 | 43.5886, 42.7940, 50.2023 |
| 1 (R3 → T0) | R3-SHM (P2P disabled / host-staged) | 44.0831 | 41.8765, 44.0831, 46.6575 |
| 1 (R3 → T0) | T0-IPC (direct P2P / NVSwitch) | 40.6836 | 30.7929, 40.6836, 48.0663 |

## Event-level max-rank percentiles

| Pass | Mode | Phase | p50 ms | p90 ms | p99 ms |
|---:|:---|:---|---:|---:|---:|
| 0 | T0-IPC (direct P2P / NVSwitch) | dispatch | 0.038144 | 0.045558 | 0.058481 |
| 0 | T0-IPC (direct P2P / NVSwitch) | combine | 0.036256 | 0.042579 | 0.055420 |
| 0 | T0-IPC (direct P2P / NVSwitch) | pair | 0.074144 | 0.088048 | 0.109816 |
| 0 | R3-SHM (P2P disabled / host-staged) | dispatch | 0.061632 | 0.080797 | 0.087718 |
| 0 | R3-SHM (P2P disabled / host-staged) | combine | 0.040752 | 0.057248 | 0.066364 |
| 0 | R3-SHM (P2P disabled / host-staged) | pair | 0.104448 | 0.126384 | 0.143519 |
| 1 | R3-SHM (P2P disabled / host-staged) | dispatch | 0.062400 | 0.080893 | 0.092258 |
| 1 | R3-SHM (P2P disabled / host-staged) | combine | 0.041088 | 0.057181 | 0.066348 |
| 1 | R3-SHM (P2P disabled / host-staged) | pair | 0.105520 | 0.127786 | 0.150193 |
| 1 | T0-IPC (direct P2P / NVSwitch) | dispatch | 0.038208 | 0.047917 | 0.068329 |
| 1 | T0-IPC (direct P2P / NVSwitch) | combine | 0.036096 | 0.044246 | 0.061675 |
| 1 | T0-IPC (direct P2P / NVSwitch) | pair | 0.073968 | 0.090032 | 0.119714 |

## Secondary timing definitions

| Definition | Pass 0 R3/T0 | Pass 1 R3/T0 |
|:---|---:|---:|
| median_max_rank_cumulative_ms | 0.9779x | 1.0162x |
| median_full_trace_cuda_interval_ms | 0.9778x | 1.0156x |
| median_wall_ms | 0.9783x | 1.0156x |

## Validation and measurement scope

- Original captured 384-event variable-size decode sequence, including rank-pair imbalance and original dispatch/combine order. Inputs and worker sources remain hash-identical during this run.
- One untimed full-trace warmup and three timed traces per mode/pass. All buffers are materialized outside timing, receives reset to NaN, and every payload element validated after each trace against its deterministic sentinel.
- The existing timing loop and aggregation are unchanged: primary = sum_event max_rank(dispatch+combine CUDA interval), median of three totals; gate = median of two pass ratios. Event percentiles pool 1,152 max-rank intervals per mode/pass. No router metadata, expert H2D/compute or cache/controller work.
- Peer bytes per trace: dispatch 139,464,704, combine 301,834,240, total 441,298,944. Self entries execute but are excluded from byte totals.
- CUDA intervals include API/launch/stream waits and are not isolated kernel durations. Whole-trace and reversed rank-max aggregation are published as secondary definitions; no threshold or timing definition was changed after observing results.
- Peak process-tree RSS 2.720 GiB; minimum host available 1852.60 GiB; minimum target GPU free 81,068 MiB. No model run or CPU Pareto rerun.
- R3 is a software-disabled P2P condition on the same NVSwitch host; it is not a physically no-NVLink server. The prior cuMem physical pilot remains historical.

## Stage boundary

Commit and stop for owner review. Clean F/K is not authorized by this result; do not add repetitions or tune transport.

See [CSV](ipc_trace_comm_replay.csv), [full result and provenance](ipc_trace_comm_replay.json), [validation](ipc_baseline_validation.json), and [execution conventions](IPC_REBASE_EXECUTION.md).
