# PAYLOAD CROSSOVER — actual MoE sizes vs T0/R3 transport

Status: authorized bounded follow-up after `fad71d2`.

## Why

The first successful non-P2P condition (R3: `NCCL_P2P_LEVEL=LOC`, `NCCL_IB_DISABLE=1`, actual `SHM/direct/direct`) was **not slower** than T0 for the tiny 32/64 KiB decode proxy.

This does not show that SHM is intrinsically faster than NVSwitch. The measured payload was small and the NCCL configurations differed strongly (T0 many P2P channels vs R3 two SHM channels). Before abandoning the H100 synthetic contrast, check only whether a latency/bandwidth crossover appears **inside the actual MoE payload range**.

Do not run the full model in this follow-up.

## Step A — actual payload distribution from an existing trace

Reuse an already captured R4/B8 raw-routing trace from the completed studies. Do not generate a new trace.

Use raw exact router selections and token origin ranks; ignore substitution decisions.

For every decode layer event compute the exact activation payload sent between each ordered rank pair for:

- dispatch;
- combine/return.

Use the runtime hidden size and BF16 element size already bound to the Qwen3-30B-A3B experiment.

Report nonzero rank-pair payload bytes:

- p50;
- p90;
- p99;
- maximum;
- fraction of messages in buckets: <=64 KiB, 64–256 KiB, 256 KiB–1 MiB, 1–4 MiB, >4 MiB.

This is analysis-only and should finish from existing artifacts.

If no suitable raw route artifact can be read deterministically, stop and report which required field is missing rather than launching the model.

## Step B — five-size crossover microbenchmark

Use exactly these **per-peer payload sizes**:

```text
32 KiB
64 KiB
256 KiB
1 MiB
4 MiB
```

Use the same four physical GPUs 0,1,4,5 and the same `all_to_all_single` API as the existing calibration.

Compare only:

### T0 — normal NVSwitch

```bash
unset NCCL_P2P_DISABLE
unset NCCL_P2P_LEVEL
unset NCCL_IB_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Expected logged data path: direct P2P/CUMEM.

### R3 — functional non-P2P SHM

```bash
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
export NCCL_IB_DISABLE=1
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Expected logged data path: `SHM/direct/direct`.

For each size/mode:

- 10 warmups;
- 30 timed iterations;
- report max-rank median and p90 CUDA interval;
- calculate effective payload bandwidth.

No H2D/concurrent test is needed here; this follow-up isolates the peer-size crossover only.

Run the two modes close in time. Use two lightweight passes to reduce temporal bias:

- pass 0: T0 -> R3;
- pass 1: R3 -> T0.

Report the median across the two pass medians. Do not add a third pass.

INFO transport logging is required only for the first size in each mode to verify the path. Disable INFO for timing.

## Interpretation

For each size report:

```text
ratio(size) = R3_peer_latency / T0_peer_latency
```

Then compare the measured crossover to the real payload distribution.

### Useful H100 synthetic contrast

Continue the original Stage 1 only if:

- R3 is >=1.5x slower than T0 at one or more sizes that cover a meaningful part of real traffic; and
- the slowdown occurs at or below the real p90/p99 payload range, not only at an artificial 4 MiB tail.

### Not useful on this H100

Stop synthetic P2P-off work and use the real no-NVLink server if:

- R3 remains <=1.5x T0 throughout the real p99 range; or
- crossover appears only at payloads larger than the observed real maximum.

Do not retune channels, protocols, message batching, or NCCL knobs to manufacture a crossover.

## Outputs

Commit only:

- `payload_distribution.csv`
- `payload_distribution.json`
- `payload_crossover.csv`
- `PAYLOAD_CROSSOVER_RESULTS.md`
- one small T0 and one small R3 INFO log proving the paths.

Commit immediately when this bounded follow-up finishes, then stop for owner review.
