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

---

## 10. 2026-10-04 amendment — broader timing-variance hypotheses

This amendment supersedes the original assumption that S2 can necessarily
compare same-NUMA and cross-NUMA GPU pairs.

S0 established only the **guest-visible** topology:
- KVM full virtualization;
- one guest NUMA node / one guest socket;
- CPUs 0--191 in node0;
- every GPU reports the same guest CPU/NUMA affinity;
- PCI NUMA node is unavailable (-1 / N/A);
- all GPUs are mutually connected by NV18.

Therefore, "guest has one NUMA node" must **not** be interpreted as proof that
the physical host has one NUMA domain or that all host PCIe paths are equivalent.
The hypervisor may flatten host topology. Conversely, PCIe/H2D contention can
exist even on a genuinely single-NUMA host through shared memory controllers,
host bridges, switches/uplinks, DMA/IOMMU resources, or other host contention.

### Hypotheses to test, in priority order

H1. **Heavy PSS monitor interference.** The current monitor scans a very large
process address space and has taken roughly 17--25 s per scan in prior runs.
It can contend with the CPU/memory/page-table activity used by expert offload.

H2. **CPU expert page residency / page-fault state.** The expert pool is
pageable and file-backed. A cold run may fault/populate pages while later runs
benefit from warm Linux page cache, changing decode time without changing model
tokens or cache-state hashes.

H3. **CPU scheduling / affinity variability.** Historically each rank could run
on CPUs 0--191. Scheduler placement and cache locality can therefore differ
between repeats. Use disjoint fixed CPU sets for the stability confirmation.

H4. **Shared H2D / PCIe / host-memory contention.** One guest NUMA node does not
rule this out. Measure identical 9-MiB H2D transfers with 1, 2, 4, and 8 GPUs
concurrent.

H5. **Env 2 SHM pair non-uniformity despite hidden NUMA.** Since the guest does
not expose physical NUMA, do not fabricate same/cross-NUMA labels. Measure an
empirical GPU-pair SHM cost matrix instead.

H6. **Hidden host NUMA / hypervisor placement.** Plausible but not directly
identifiable from the guest. Treat it only as an explanation for measured pair
or H2D non-uniformity unless host-level topology becomes available.

Lower-priority / largely screened hypotheses:
- thermal throttling: prior discrepant runs both sampled max 41 C;
- recompilation: no-compilation checks passed;
- different computation: token and final-cache hashes matched.

### S2 replacement — host-memory and transport diagnostics

Finish the already-started six-run S1 sequence unchanged. Then use the following
S2 instead of the original same-NUMA/cross-NUMA requirement.

#### S2A — expert-pool residency A/B

Reuse the same frozen P/CA-rep decode64 workload with BOUNDARY monitoring.

Compare:
1. **NORMAL**: current pageable file-backed expert pool behavior.
2. **PRETOUCH**: before the timed region, touch/read every CPU expert page that
   the frozen schedule can access so the pages are resident in host memory.
   Do not copy experts to GPU and do not change GPU-cache state.

Use counterbalanced order:
```
NORMAL, PRETOUCH, PRETOUCH, NORMAL
```

Record before/after:
- major/minor page-fault counters for each rank or process tree;
- host MemAvailable/Cached;
- decode wall and E2E;
- token/state hashes.

Do not drop the system page cache and do not require sudo.

Interpretation: if PRETOUCH materially reduces both variance and page faults,
mark `PAGE_RESIDENCY_CONFOUND`. Do not infer disk I/O unless major faults or
I/O evidence supports it; file-backed does not automatically mean disk access.

#### S2B — concurrent 9-MiB H2D scaling

With no model and no heavy monitor, transfer the same 9-MiB payload:
- 1 GPU;
- 2 GPUs concurrently;
- 4 GPUs concurrently;
- 8 GPUs concurrently.

Use 20 warmups + 50 measured iterations per level. Record aggregate and
per-GPU median/p90 bandwidth and slowdown relative to the 1-GPU baseline.

This diagnoses shared PCIe/host-memory/DMA contention without making a NUMA
claim.

#### S2C — Env 2 empirical SHM pair matrix

Under validated Env 2 (P2P disabled, SHM path), measure all unordered GPU pairs
if practical; otherwise at minimum cover every GPU as both a low/high-ID
endpoint. Payloads remain exactly:
```
32 KiB, 128 KiB, 512 KiB
```

Use 20 warmups + 50 measured iterations per pair/payload. Report median/p90 and
the max/min pair ratio. Label pairs only by GPU IDs and measured cost; do not
call any pair same-NUMA or cross-NUMA from guest information.

If pair costs are materially non-uniform, preserve the measured matrix as the
transport topology for later communication-aware placement analysis.

### S3 amendment

For the physical stability confirmation:
- use BOUNDARY monitoring;
- use disjoint fixed CPU affinity per rank;
- use the PRETOUCH behavior only if S2A establishes page-residency instability,
  and then apply it identically to Env 1 and Env 2;
- keep all other model/cache/policy semantics unchanged.

The existing <=5% spread gate remains unchanged. No policy comparison resumes
until this gate passes.

