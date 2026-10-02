# PHYSICAL F/K/C PILOT — does the preferred replica budget move with transport?

Status: authorized after CPU screen commit `dc7b099`.

## Question

The CPU replay established a strong byte-level frontier:

- F: rho=0.00 — minimum H2D;
- K: rho=0.25 — intermediate knee;
- C: rho=0.75 — minimum peer communication.

This pilot asks only:

> When the **same frozen residency actions** are executed physically, does the preferred rho region differ between normal NVSwitch P2P (T0) and the validated non-P2P SHM path (R3)?

This is a six-cell **descriptive pilot**, not final performance evidence.

## Scope

Use physical GPUs **0,1,4,5** only.

Fixed:

- Qwen3-30B-A3B-Instruct-2507 BF16;
- R4, local B8 / global batch 32;
- cache ratio 30%;
- exact experts only;
- substitution OFF;
- one prefill + **8 decode forwards**;
- same prompts/input ordering as the validated exact-only capture;
- F/K/C = rho 0 / 0.25 / 0.75;
- one physical timing run per transport/point;
- no load objective;
- no migration;
- no D2D full-expert copy;
- every newly created physical expert copy comes from the CPU expert store through the normal H2D path.

Do not extend to 32/64 decode, B4/B16, R8 or more rho values in this packet.

## Freeze the CPU policy before GPU timing

Regenerate the deterministic CPU replay from `dc7b099` and emit one immutable action schedule for each of F/K/C.

Each event schedule must contain at least:

- mandatory first-copy admissions;
- replica admissions;
- victim evictions;
- resulting physical owner/copy set;
- route destination for every raw exact route;
- dispatch/combine send-count matrices;
- expected H2D fetch class/count;
- post-event cache-state hash.

Hash the three schedules and commit their compact metadata/provenance.

During physical timing **do not recompute the greedy replica policy**. The GPU runtime only applies the frozen event actions. Policy search/planning time is therefore outside the primary timing.

A narrow experiment-only replay path is preferred over changing the production controller semantics.

## Trace parity gate

At every physical event require:

1. raw selected expert IDs equal the validated exact-only source trace;
2. token-origin ranks equal the source trace;
3. all selected experts execute exactly;
4. applied owner/copy state equals the frozen schedule;
5. dispatch/combine send counts equal the frozen schedule;
6. no rank exceeds physical cache capacity;
7. generated tokens remain identical across all six cells.

If raw routing diverges at any event, stop that cell and report `TRACE_DIVERGENCE`; do not silently continue with a new trajectory.

## Transport conditions

### T0 — normal NVSwitch P2P

Clear transport overrides:

```bash
unset NCCL_P2P_DISABLE
unset NCCL_P2P_LEVEL
unset NCCL_IB_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Use the already validated T0 setup. Timing must run with NCCL INFO disabled.

### R3 — non-P2P SHM

Use exactly:

```bash
unset NCCL_P2P_DISABLE
export NCCL_P2P_LEVEL=LOC
export NCCL_IB_DISABLE=1
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_C2C
unset NCCL_SHM_DISABLE
```

Expected path remains `SHM/direct/direct`.

Do not search any more NCCL knobs. A single tiny preflight path check per mode is allowed only if needed to verify that the environment still maps to the previously validated transport; exclude it from timing.

## Six-cell order

Run fresh processes in this exact interleaved/counter-directed order:

1. T0-F
2. R3-C
3. T0-K
4. R3-K
5. T0-C
6. R3-F

Thus T0 traverses F -> K -> C while R3 traverses C -> K -> F, reducing simple time-order bias.

One run per cell only. Do not add repeats automatically.

## Primary timing

Exclude model load, schedule generation and transport preflight.

Record:

- prefill duration;
- total eight-step decode duration;
- mean decode ms/step;
- short-run E2E generation duration;
- max-rank replay/controller application CPU time.

The primary comparison is **decode duration / mean decode ms per step**, not model-load wall time.

Because this is a single-run pilot, report raw values and relative deltas but do not attach confidence intervals or claim a stable speedup.

## Mechanism counters

For every cell collect the physical counterparts of the CPU-screen axes:

- expert H2D fetch count;
- first-copy / reload / replica fetch counts;
- expert H2D bytes;
- peer activation bytes;
- remote token-rank pairs;
- mean/peak duplicate-slot fraction;
- mean unique residents.

Also collect lightweight runtime timing if already supported without a profiler:

- cumulative/max-rank expert H2D interval;
- cumulative/max-rank NCCL/peer interval;
- expert GPU kernel time;
- layer/MoE completion interval.

Do **not** add Nsight runs in this pilot. If a counter would require invasive profiling, omit it and say so rather than expanding scope.

## What would be interesting

For each transport define the descriptively fastest point:

```text
rho*_T = argmin_{rho in {0,.25,.75}} decode_time(T,rho)
```

The strongest motivation for a transport-aware method would be:

```text
rho*_T0 != rho*_R3
```

with mechanism counters consistent with the reason:

- moving toward C increases physical H2D work;
- moving toward C reduces physical peer communication;
- R3 assigns more time penalty to the remaining communication than T0.

Do not force this outcome.

## Pilot interpretation

### PROMISING_SHIFT

Report this only if:

- all six correctness/trace gates pass;
- the descriptively fastest rho differs between T0 and R3; and
- the winning point beats the next-best point in that transport by at least **5%** in decode time; and
- H2D/peer counters move in the expected directions.

This authorizes owner consideration of a repeated physical validation; it is not itself a final speedup claim.

### NO_CLEAR_SHIFT

Report this if:

- both transports prefer the same rho; or
- the apparent winner margin is <5%; or
- component timings do not support the observed total-time ordering.

Do not tune rho or alter policy after seeing the result.

### FAIL_CORRECTNESS

Any route/token/cache/action mismatch stops the affected cell and the packet.

## Guardrails

- Check GPUs 0,1,4,5 are free before launch.
- Keep the existing host-memory guard.
- Stop on OOM rather than reducing the model/cache.
- Do not alter F/K/C after seeing timing.
- Do not enable substitution.
- Do not implement a weighted objective or final online selector.
- Do not add repeats, longer decode, Nsight, or the real no-NVLink server automatically.

## Outputs

Commit:

- `physical_fkc_pilot.csv`
- `physical_fkc_pilot.json`
- `PHYSICAL_FKC_RESULTS.md`
- `physical_fkc_validation.json`
- compact frozen-schedule metadata/hashes;
- small transport/path receipts only if a new preflight was necessary.

Commit after the six-cell pilot (or immediately on a blocking failure) and stop for owner review.
