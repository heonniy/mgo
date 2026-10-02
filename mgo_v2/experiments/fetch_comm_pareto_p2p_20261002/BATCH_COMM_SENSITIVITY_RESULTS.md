# Batch communication sensitivity — B4 through B32

**B4→B32: BATCH_SENSITIVE**; original B4→B16 endpoint: **MIXED**.

The full batch curve is nonmonotonic: the endpoint rule is a descriptive screen, not evidence of a robust bandwidth crossover. B4 itself reverses direction between passes, and B8 exceeds both B16 and B32. The B32 endpoint increases the ratio by 10.62 percentage points, only 0.62 points beyond the predeclared 10-point threshold. No extra repetitions were added.

Local batches 4/8/16/32 correspond to global 16/32/64/128 on R4. Cache30, exact-only P0 seed42/LRU, no replicas. B8 source reused; exactly three new diagnostic model captures. All captures/counts and communication payloads passed. Model capture times are not performance evidence.

## Message geometry

| Local batch | Phase | Peer MiB | Nonzero messages | Message p50/p90/p99 KiB | Self fraction |
|---:|:---|---:|---:|:---|---:|
| 4 | dispatch | 66.836 | 4602 | 16.0 / 16.0 / 16.0 | 0.250 |
| 4 | combine | 144.070 | 4602 | 32.0 / 48.0 / 60.0 | 0.250 |
| 4 | union | 210.906 | 9204 | 16.0 / 40.0 / 56.0 | 0.250 |
| 8 | dispatch | 133.004 | 4608 | 32.0 / 32.0 / 32.0 | 0.250 |
| 8 | combine | 287.852 | 4608 | 64.0 / 88.0 / 112.0 | 0.250 |
| 8 | union | 420.855 | 9216 | 32.0 / 80.0 / 108.0 | 0.250 |
| 16 | dispatch | 263.422 | 4608 | 60.0 / 64.0 / 64.0 | 0.250 |
| 16 | combine | 576.438 | 4608 | 128.0 / 180.0 / 232.0 | 0.249 |
| 16 | union | 839.859 | 9216 | 64.0 / 164.0 / 220.0 | 0.249 |
| 32 | dispatch | 528.715 | 4608 | 124.0 / 128.0 / 128.0 | 0.250 |
| 32 | combine | 1151.047 | 4608 | 252.0 / 364.0 / 463.7 | 0.251 |
| 32 | union | 1679.762 | 9216 | 128.0 / 324.0 / 436.0 | 0.250 |

## Whole-trace primary timings

| Local batch | CUDA R3/T0 pass 0 / pass 1 | Median CUDA ratio | Median wall ratio |
|---:|:---|---:|---:|
| 4 | 1.2767 / 0.9799 | 1.1283 | 1.1283 |
| 8 | 1.8385 / 1.3445 | 1.5915 | 1.5906 |
| 16 | 0.9027 / 0.9844 | 0.9435 | 0.9434 |
| 32 | 1.2020 / 1.2670 | 1.2345 | 1.2345 |

Per-repeat whole CUDA/wall times, both secondary aggregation definitions and dispatch/combine/pair p50/p90/p99 are retained in the timing CSV/JSON. Primary is max rank whole-trace interval per repeat, median of three; reported ratio summarizes two counter-ordered passes. Event sum-max is diagnostic only.

## Descriptive interpretation

- B4_to_B16: MIXED; union message median 4.000x, median max-rank remote bytes 4.047x, nonzero message count 1.001x. CUDA ratio changes -18.48 percentage points; wall ratio -18.49 points.
- B4_to_B32: BATCH_SENSITIVE; union message median 8.000x, median max-rank remote bytes 8.070x, nonzero message count 1.001x. CUDA ratio changes +10.62 percentage points; wall ratio +10.63 points.

These are bounded descriptive measurements, not confidence-tested causal evidence. Event intervals include launch/stream scheduling gaps. Whole-trace throughput and event-level latency can differ; the older B8 IPC/SHM result is historical context, not pooled into this run.

## Calibration and event diagnostics

- T0: alpha=0.122679 ms, beta=-1.013e-07 ms/byte, R²=0.153. Five per-peer sizes only; no physical bandwidth claim.
- R3: alpha=0.062947 ms, beta=7.747e-08 ms/byte, R²=0.318. Five per-peer sizes only; no physical bandwidth claim.

The calibration fits are weak: T0 has a negative byte slope and both R² values are below 0.32. These five-size measurements do not identify a reliable positive byte-transfer cost or a clean latency/bandwidth crossover.

[Event buckets](batch_comm_event_buckets.csv) report pair median/p90, R3/T0, unique event fractions and byte fractions for each batch and pooled. [Diagnostics](batch_comm_event_diagnostics.json) include correlations with bytes and fan-out; constant predictors are null. Predictors are transport-independent and fixed before timing.

## Validation and limits

- Peak process-tree RSS 228.91 GiB; minimum host available 1620.17 GiB; minimum target GPU free 68,385 MiB. No memory guard stop.
- All CPU Pareto results and frozen schedule metadata retain their SHA256 hashes. No CPU screen, F/K/C run, additional decode, NCCL tuning, or NVLink-mode operation.
- Fresh transport smokes accepted T0 P2P/IPC and R3 SHM only. Both use NCCL_CUMEM_ENABLE=0. INFO is absent during timing.
- Eight resident-model inference workers are paused throughout experiment stages and restarted on exit. Their safeguards yield to other GPU jobs and memory/temperature pressure. Idle inference is separate from the three research captures.
- Raw capture/timing receipts are outside Git with hashes; compact geometry, prompt provenance, diagnostics and results are checked in. No automatic F/K model timing; stop for owner review.

See [geometry](batch_comm_geometry.json), [trace timing](batch_comm_trace_timing.json), [calibration](small_payload_latency.json), [validation](batch_comm_validation.json), and [frozen conventions](BATCH_COMM_EXECUTION.md).
