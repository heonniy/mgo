# NVLINK BANDWIDTH LADDER — FULL vs OFF/PCIe-P2P vs SHM

Status: owner-authorized after `3e59975`.

## Motivation

The stable IPC actual-trace replay measured only a small whole-trace transport
gap between:

- T0-IPC direct P2P/NVSwitch; and
- R3-SHM P2P-disabled host-staged communication.

The two pass ratios were 1.0529x and 1.0836x (median 1.0682x), so clean F/K
model timing was correctly not launched.

The next question is narrower:

> What does the same MoE communication trace cost when NVLink bandwidth is
> explicitly disabled but CUDA/NCCL direct P2P is retained, i.e. a PCIe-P2P
> condition, relative to FULL NVSwitch and P2P-disabled SHM?

This is a transport characterization only. It does not change the CPU Pareto
frontier or authorize a new controller.

## CPU results remain frozen

Do **not** rerun the CPU replica screen.

The existing CPU results depend only on frozen routing/residency decisions and
counts (H2D bytes, peer bytes, duplicates, coverage). They contain no measured
transport latency. Therefore FULL/OFF/SHM changes only the physical cost
assigned to the same peer bytes.

F/K/C and all CPU hashes remain unchanged.

---

## Safety gate — global NVLink state

`nvidia-smi nvlink -sBwMode` may change NVLink bandwidth mode beyond the four
target GPUs. Treat it as a **server-global intervention** for this study.

Before any write:

1. query supported syntax/modes read-only;
2. query current bandwidth mode;
3. save `nvidia-smi` process/utilization state for all 8 GPUs;
4. require **all eight GPUs to have zero compute processes and <1 GiB used**;
5. require the current bandwidth mode to be FULL.

If any condition fails, record `BLOCKED_SERVER_BUSY`,
`BLOCKED_UNSUPPORTED`, or `BLOCKED_PRIVILEGE` and stop without changing
the mode.

Do not use sudo, prompt for a password, kill jobs, or wait indefinitely for
other users.

### Mandatory restoration

Any code path that successfully changes bandwidth mode to OFF must install a
finally/trap cleanup that attempts:

```bash
nvidia-smi nvlink -sBwMode FULL
```

before exit.

After restoration, query the mode again and require FULL. If restoration
cannot be verified, stop everything, emit `RESTORE_FAILED`, and make that the
top-level result. Do not launch any further communication or model work.

---

## Common NCCL memory-sharing setting

The default P2P/CUMEM path is known unstable on this host. For all three
conditions use:

```bash
export NCCL_CUMEM_ENABLE=0
```

so the comparison does not reintroduce the cuMem mapping failure.

No CUMEM debugging is part of this packet.

---

# Condition A — FULL NVSwitch direct P2P

Require NVLink bandwidth mode FULL.

NCCL environment:

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

- all-rank payload validation passes;
- NCCL peer path is `P2P/IPC`;
- no SHM/NET path for measured peer transfers.

Label:

```text
FULL-P2P = NVSwitch FULL + P2P/IPC
```

---

# Condition B — NVLink bandwidth OFF, direct PCIe P2P

Only after the global safety gate passes, execute the supported OFF command:

```bash
nvidia-smi nvlink -sBwMode OFF
```

Immediately query bandwidth mode and require OFF before launching NCCL.

Use the **same NCCL environment as Condition A** so direct P2P remains enabled.

Acceptance:

- bandwidth-mode query reports OFF;
- all-rank payload validation passes;
- NCCL peer path is still `P2P/IPC`;
- no SHM/NET path for measured peer transfers.

Interpret this condition only as:

```text
OFF-P2P = NVLink bandwidth OFF + direct GPU P2P, expected to fall back to PCIe P2P
```

Do not claim a particular PCIe route from the NCCL transport string alone.
The physical route is characterized by the NVLink bandwidth-mode state plus
measured performance.

After all OFF measurements finish, restore FULL **before** Condition C.

---

# Condition C — FULL hardware, P2P-disabled SHM

First verify that NVLink bandwidth mode is restored to FULL.

NCCL environment:

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

- all-rank payload validation passes;
- peer path is SHM;
- no P2P/NET path.

Label:

```text
SHM = P2P disabled / host-staged fallback
```

The host still physically has NVSwitch; this is not called a physical
no-NVLink machine.

---

# Stage L0 — tiny payload calibration

Run this only after each condition's path/state smoke passes.

Use fixed per-peer BF16 payloads representative of the measured MoE range:

```text
32 KiB
128 KiB
```

For each condition and size:

- 10 warmups;
- 30 timed iterations;
- report median and p90 max-rank collective time;
- validate payloads;
- no model.

This is 3 conditions x 2 sizes = 6 tiny calibration cells.

Purpose: verify the expected ordering without using large-message bandwidth as
a proxy for the real MoE workload.

No automatic tuning if ordering is surprising.

---

# Stage L1 — same 384-event actual MoE communication trace

Reuse the exact frozen matrices and worker conventions already validated in the
IPC E1 result.

For each condition:

- one untimed full-trace warmup;
- exactly three timed full-trace replays;
- same original 384 decode-layer events;
- same dispatch/combine order and variable split sizes;
- all buffers allocated outside timing;
- payload validation outside timing;
- no model, expert H2D, cache/controller, or router collectives;
- INFO disabled during timing.

Execution order:

```text
1. FULL-P2P
2. OFF-P2P
3. restore FULL and verify
4. SHM
```

Because bandwidth-mode mutation is global, do not repeatedly toggle it just to
counterbalance order. Instead report this as a bounded characterization and
retain the prior `3e59975` FULL/SHM measurements as historical replication
context.

Primary values:

```text
T_FULL
T_OFF
T_SHM
OFF/FULL
SHM/FULL
SHM/OFF
```

Report both:

- median cumulative primary trace metric already used by E1; and
- whole-trace CUDA/wall secondary metrics.

Also report event p50/p90/p99 for dispatch, combine, and pair.

---

# Interpretation

This stage is descriptive; there is **no automatic F/K model run**.

Useful outcomes:

### REALISTIC_INTERMEDIATE

```text
T_FULL < T_OFF < T_SHM
```

with OFF/FULL >= 1.10.

This supports a three-level communication-cost ladder and authorizes owner
review for a later physical F/K test on FULL versus OFF.

### OFF_CLOSE_TO_FULL

OFF/FULL < 1.10.

This says PCIe-P2P-like communication is still too cheap for this small-message
MoE trace to materially separate from NVSwitch at this workload.

### OFF_NOT_INTERMEDIATE

OFF is not between FULL and SHM, or path/state checks fail.

Report the measured result without retuning NCCL. Do not manufacture a slower
condition.

The previous SHM/FULL ~1.068 result already indicates that a large latency
shift is not guaranteed.

---

# Required outputs

- `nvlink_bw_ladder.csv`
- `nvlink_bw_ladder.json`
- `NVLINK_BW_LADDER_RESULTS.md`
- `nvlink_bw_ladder_validation.json`
- compact FULL/OFF/SHM transport receipts;
- pre-change and post-restore all-GPU snapshots.

Do not commit giant logs.

## Hard boundaries

- no model generation;
- no CPU Pareto rerun;
- no F/K/C replay;
- no HALF/MIN/3QUARTER sweep;
- no Nsight;
- no NCCL channel/protocol tuning;
- no driver reset/reboot;
- no sudo/password prompt;
- no experiment if any of 8 GPUs is occupied;
- always restore FULL after OFF;
- commit result and stop for owner review.
