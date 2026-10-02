# IPC BASELINE REBASE — stable direct-P2P T0 vs non-P2P SHM R3

Status: owner-authorized after `f64eb56`.

## Decision

The default NCCL `P2P/CUMEM` path is not usable on this host for the current study:

- 3/3 fresh default T0 preflights timed out;
- `NCCL_CUMEM_ENABLE=0` selected `P2P/IPC` and passed four-rank payload validation.

Therefore the physical transport comparison is rebased to a stable pair:

- **T0-IPC**: direct GPU P2P over NVLink/NVSwitch, NCCL transport `P2P/IPC`;
- **R3-SHM**: direct GPU P2P disabled, NCCL SHM fallback.

The research contrast remains:

```text
fast direct GPU P2P / NVSwitch
vs
non-P2P host-staged SHM
```

It is **not** a comparison of cuMem versus IPC APIs.

## CPU Pareto result is unchanged

Do **not** rerun the CPU replica Pareto screen.

The CPU screen at `dc7b099` depends only on the frozen routing/residency replay and counts:

- expert H2D fetches/bytes;
- peer activation bytes;
- replica budget / cache coverage.

It does not execute NCCL and does not use a transport-specific latency weight.

Therefore changing T0 from `P2P/CUMEM` to `P2P/IPC` does not change any raw CPU Pareto point, F/K/C selection, H2D byte count, or peer-byte count.

Only later **physical time assigned to the same peer bytes** can change.

## Matched transport environment

Set `NCCL_CUMEM_ENABLE=0` in **both** transport conditions so the unstable device cuMem path is excluded from the comparison.

### T0-IPC — direct P2P enabled

```bash
export NCCL_CUMEM_ENABLE=0
unset NCCL_P2P_DISABLE
unset NCCL_P2P_LEVEL
unset NCCL_IB_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Acceptance:

- communicator is one local node / four local ranks;
- payload validation passes;
- channel lines use `P2P/IPC`;
- no `via SHM/` or `via NET/` is used for measured peer transfers.

This is the new T0 for all new physical timing in this packet.

### R3-SHM — direct P2P disabled

```bash
export NCCL_CUMEM_ENABLE=0
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
export NCCL_IB_DISABLE=1
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Acceptance:

- communicator is one local node / four local ranks;
- payload validation passes;
- measured peer channel lines use `via SHM/`;
- no `via P2P/` or `via NET/` is used.

Do not require the exact SHM suffix string if NCCL prints a different SHM submode; the key condition is SHM transport with no direct P2P/NET.

## Step I0 — paired smoke

Before E1, run one fresh 32-KiB four-rank smoke for each mode:

1. T0-IPC;
2. R3-SHM.

INFO logging only for these two smokes.

If either fails, commit `IPC_REBASE_BLOCKED` and stop. Do not search more NCCL knobs.

## Step I1 — re-run actual-trace communication replay

If both smokes pass, immediately run the already frozen 384-event decode communication replay.

Use the same real dispatch/combine matrices, event order and measurement conventions from `TRACE_COMM_REPLAY_PROTOCOL.md`.

Counter-order:

```text
pass 0: T0-IPC -> R3-SHM
pass 1: R3-SHM -> T0-IPC
```

For each mode/pass:

- one untimed full-trace warmup;
- three timed full-trace replays;
- no model;
- no expert H2D;
- no cache/controller work;
- no INFO during timing.

Use the existing E1 decision thresholds unchanged:

- **STRONG_GAP**: R3 slower in both passes and median R3/T0 ratio >=1.20;
- **NO_GAP** / **AMBIGUOUS_GAP** as already defined.

Commit I0/I1 results before any model run.

## Step I2 — clean F/K only, conditional

Only if I1=STRONG_GAP:

- use the existing frozen F=rho0 and K=rho0.25 schedules;
- retain the clean timed-path requirements from `TRACE_COMM_CLEAN_FK.md`;
- run two repeats per point/transport;
- use T0-IPC and R3-SHM exactly as defined above.

The old six-cell F/K/C physical pilot remains historical because its T0 transport backend differed and its timed path included heavy validation.

Do not rerun C.

## Naming in results

Use these labels:

- `T0-IPC (direct P2P / NVSwitch)`
- `R3-SHM (P2P disabled / host-staged)`

Do not call R3 a physically no-NVLink machine. The host still has NVSwitch hardware; direct P2P is disabled by software.

## Hard boundaries

- no CPU Pareto rerun;
- no change to rho or F/K selection;
- no more cuMem debugging;
- no use of default `P2P/CUMEM` in new timing;
- no NCCL channel/protocol tuning;
- no substitution;
- no C point;
- no E2 unless actual-trace I1 is STRONG_GAP.
