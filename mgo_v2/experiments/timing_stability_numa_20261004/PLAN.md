# PLAN — Timing stability before policy timing

Date: 2026-10-04
Status: owner-authorized diagnostic.
Priority: **pause the physical Env 1 / Env 2 policy matrix**.

## 1. Goal

Before interpreting any BR / CA / CA-rep E2E or TPOT result, establish that the
same frozen physical offloading workload repeats stably.

No policy conclusion is allowed from the existing noisy timing samples.

Primary stability target:

```
(max_time - min_time) / median_time <= 5%
```

for three clean repeats of the same workload. <=3% is preferred.

The existing compile artifacts, frozen schedule, model checkpoint and request
manifest are reused. No new scientific policy sweep.

---

## 2. Freeze one diagnostic workload

Reuse the existing **P / CA-rep / Env 1** frozen schedule, because that exact
cell exhibited the 605 s vs 466 s discrepancy.

For diagnosis only:
- use the exact first **64 decode steps** of the frozen 256-step schedule;
- same model/runtime/cache ratio/Gate/substitution state;
- same request ordering;
- same action schedule;
- same output-token/state hash validation;
- no policy recomputation inside timing.

Only after the 64-step harness passes do we run a 256-step stability confirmation.

---

## 3. S0 — topology and process inventory (no GPU timing)

Record:
- `nvidia-smi topo -m`;
- GPU -> NUMA node mapping;
- `numactl -H`;
- CPU core lists per NUMA node;
- target-rank CPU affinity before launch;
- `MemAvailable`;
- foreign GPU processes;
- current CPU load / runnable-task count.

Do not infer topology from GPU IDs. Save the actual mapping.

Stop the project's resident load workers before diagnosis and verify their GPU
memory is released.

---

## 4. S1 — isolate monitor interference

Compare exactly two measurement modes on the same decode64 workload.

### HEAVY
The existing monitor unchanged:
- process-tree PSS / memory-map traversal;
- current scan cadence.

### BOUNDARY
No concurrent safety-monitor process during the timed region.

Perform the same safety checks immediately before and after timing:
- GPU temperature;
- GPU free memory;
- foreign GPU compute processes;
- host MemAvailable;
- STOP marker.

No PSS, `nvidia-smi`, `ps`, `/proc/*/smaps`, or memory-map walk may execute
concurrently with the BOUNDARY timed region.

Run six total measurements, counterbalanced:

```
HEAVY, BOUNDARY, BOUNDARY, HEAVY, HEAVY, BOUNDARY
```

Before every run:
- fresh worker process;
- compiled artifacts already present;
- one untimed warmup;
- reset frozen cache/request/RNG state;
- GPU synchronize;
- same CPU affinity.

Record only boundary E2E/decode timing in the timed worker.

### S1 interpretation

Call `MONITOR_CONFOUND` if:
- BOUNDARY spread <=5%; and
- HEAVY median is >=10% slower than BOUNDARY **or** HEAVY spread is at least
  2x BOUNDARY spread.

If BOUNDARY itself is >5% unstable, do not blame the monitor; proceed to NUMA
and host-placement diagnosis.

Regardless of outcome, all future scientific MEASURE runs must use the lowest
overhead mode that passes safety requirements; expensive PSS belongs before/
after MEASURE or in the separate COUNTERS pass.

---

## 5. S2 — NUMA characterization

This stage is diagnostic and has no model execution.

### 5.1 H2D locality

For each NUMA domain that owns experiment GPUs:
- allocate/pin a 9-MiB host expert buffer on the GPU-local NUMA node;
- measure H2D to its local GPU;
- repeat with the host buffer pinned to the other NUMA node.

Use 20 warmups + 50 measured copies. No concurrent monitor.

Report local/remote median and p90.

### 5.2 Env 2 SHM locality

Using the topology from S0, choose:
- one same-NUMA GPU pair;
- one cross-NUMA GPU pair.

Under Env 2 only (P2P disabled, SHM validated), measure representative
dispatch+return payloads:

```
32 KiB, 128 KiB, 512 KiB
```

20 warmups + 50 measured iterations per payload/pair. Pin each rank process to
CPU cores local to its GPU. No model, no expert H2D, no PSS monitor.

The purpose is only to determine whether cross-NUMA SHM is materially slower
than same-NUMA SHM on this host.

Do not call Env 2 "PCIe-only".

---

## 6. S3 — fixed-affinity physical stability

After S0/S1/S2, freeze a deterministic affinity policy:
- each rank process bound to CPU cores local to its GPU;
- CPU expert-memory allocation follows the intended local NUMA placement where
  runtime support allows it;
- no CPU-core overlap between model ranks;
- no concurrent heavy monitor.

Run the same frozen P/CA-rep decode64 workload:
- Env 1: 3 repeats;
- Env 2: 3 repeats.

Counterbalance environment order:
```
Env1, Env2, Env2, Env1, Env1, Env2
```

Accept `HARNESS_STABLE` only if each environment independently has <=5%
decode-time spread and all hash/state checks pass.

If stable, run exactly three **decode256** repeats of the same P/CA-rep cell in
the environment(s) needed to confirm the fix. The 256-step confirmation must
also meet <=5% spread.

If instability remains >5%, stop. Do not resume BR/CA/CA-rep timing and do not
select the fastest repeat.

---

## 7. Timing hygiene

All diagnostic timed runs reuse the existing separation:

```
PLAN (untimed) -> COMPILE (already complete/untimed) -> MEASURE -> COUNTERS
```

During MEASURE:
- no compilation;
- no policy optimization;
- no per-layer/per-token logging;
- no Nsight;
- no detailed hit/miss counters;
- no PSS/smaps walk;
- only outer boundary timing.

Detailed counters remain separate and are not rerun unless needed for hash/
state validation.

---

## 8. Required outputs

### Monitor table
- mode;
- repeat;
- E2E;
- decode wall;
- TPOT;
- monitor scan time if HEAVY;
- token/state hashes.

### NUMA table
- GPU pair;
- same/cross NUMA;
- payload;
- Env 2 SHM median/p90;
- H2D local/remote median/p90.

### Stability table
- Env;
- affinity;
- repeats;
- median/min/max;
- spread %;
- PASS/FAIL.

Also record exact CPU affinity and memory-binding commands so later policy
experiments reproduce the same environment.

---

## 9. Scope boundary

Until `HARNESS_STABLE`:
- no new BR vs CA timing;
- no new CA vs CA-rep timing;
- no new sample/seed search timing;
- no broader cache/batch/R sweep;
- no performance claim from existing noisy samples.

Preserve all completed policy checkpoints but mark them timing-invalid for
comparison until the harness is validated.

After this packet, stop for owner review before resuming the physical matrix.
