# CUMEM PREFLIGHT RETRY — bounded diagnosis before re-running E1

Status: authorized after blocked preflight commit `b71199c`.

## Why

The first E1 T0 preflight reached:

- NCCL communicator initialization complete;
- T0 selected `P2P/CUMEM`;
- shareable-buffer import / UDS handle mapping;

but produced no validated payload receipt before the 180 s bound.

The 384-event trace input and CPU tests already pass. This retry packet is only to determine whether the failure was a transient P2P/CUMEM/UVM mapping stall and, if the default T0 path is stable again, to re-run the original E1 measurement unchanged.

Do not change the research method in this packet.

## D0 — environment snapshot

Before each diagnostic smoke, record but do not modify:

- `nvidia-smi` process/utilization snapshot for all eight GPUs;
- target GPU free memory for 0,1,4,5;
- host available memory;
- `NCCL_*` environment;
- CUDA/PyTorch/NCCL versions.

Do not kill or disturb workloads on GPUs 2,3,6,7.

Target GPUs 0,1,4,5 must each have <1 GiB used before launch.

## D1 — default T0 stability check

Run the same tiny 32-KiB T0 smoke in **three fresh process groups** with no transport override.

Environment:

```bash
unset NCCL_P2P_DISABLE
unset NCCL_P2P_LEVEL
unset NCCL_IB_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
unset NCCL_CUMEM_ENABLE
```

Requirements:

- fresh process group every trial;
- same GPUs 0,1,4,5;
- INFO logging enabled only for the smoke;
- require `P2P/CUMEM` in the channel lines;
- require four-rank payload validation;
- bound each trial at **90 seconds**.

Record PASS / TIMEOUT / FAIL independently. A timeout is a failure, not a slow sample.

If a trial stalls, collect:

- last 50 NCCL lines per rank;
- worker wait channels when available;
- process/GPU memory at timeout;

then clean up and continue to the next fresh trial. No driver reset.

## D2 — one diagnostic with cuMem disabled

Run exactly one additional fresh T0 smoke with:

```bash
export NCCL_CUMEM_ENABLE=0
```

and otherwise the same default T0 environment.

This trial is **diagnostic only**. Record the actual selected P2P transport string; do not assume IPC in advance.

Use the same 90 s bound and payload validation.

Do not use `NCCL_CUMEM_ENABLE=0` as the experiment baseline without a later owner decision.

## Diagnosis

Classify:

### TRANSIENT_RECOVERED

All three default D1 trials PASS on `P2P/CUMEM`.

The prior `b71199c` timeout is treated as an isolated/transient infrastructure stall for this study. The D2 result is recorded only as diagnosis.

**Then re-run original E1 automatically**, using the original default T0 and original R3 environments exactly as defined in `TRACE_COMM_CLEAN_FK.md`.

Use a fresh result root; do not overwrite the failed `trace_comm_20261003` receipts.

Commit the D1/D2 diagnosis before launching E1.

### CUMEM_PATH_UNSTABLE

At least one default D1 trial TIMEOUT/FAIL, while D2 with `NCCL_CUMEM_ENABLE=0` PASSes.

This supports a cuMem-path-specific instability diagnosis, but does not establish the exact driver root cause.

Stop after committing diagnosis. Do **not** run E1 with cuMem disabled and do not reinterpret that alternate path as T0.

### BROADER_P2P_UVM_FAILURE

Default D1 is unstable and D2 also TIMEOUT/FAIL.

Stop after diagnosis. Do not search more NCCL knobs or reset the driver.

### ALTERNATE_PATH_FAILURE

D1 passes 3/3 but D2 fails.

Record it. Because the production/default T0 path is stable, this does not block re-running original E1.

## E1 re-run after TRANSIENT_RECOVERED only

Re-run Stage E1 exactly as previously frozen:

```text
pass 0: T0 -> R3
pass 1: R3 -> T0
```

For each cell:

- one untimed full-trace warmup;
- three timed full-trace replays;
- identical 384-event dispatch/combine matrices;
- no model / expert H2D / cache controller;
- no NCCL INFO during timing.

Do not change the STRONG_GAP / NO_GAP thresholds.

If E1 returns STRONG_GAP, stop after committing E1. The clean F/K model stage still follows the existing conditional authorization and must use the cleaned timing path; do not merge diagnosis and model timing into one commit.

## Hard boundaries

Do not:

- raise a hung smoke timeout above 90 s;
- retry more than three default + one cuMem-disabled diagnostic smoke;
- change NCCL channels/protocols;
- use `NCCL_CUMEM_ENABLE=0` for timed E1;
- reset NVIDIA drivers;
- reboot the server;
- touch other users' GPU jobs;
- start E2 unless the original E1 STRONG_GAP gate passes.

## Outputs

Commit diagnosis artifacts:

- `cumem_preflight_retry.csv`
- `cumem_preflight_retry.json`
- `CUMEM_PREFLIGHT_RETRY_RESULTS.md`
- compact per-trial NCCL excerpts;
- environment/GPU snapshots.

If D1 is 3/3 PASS, commit this diagnosis first and then re-run E1 into a fresh result directory.
