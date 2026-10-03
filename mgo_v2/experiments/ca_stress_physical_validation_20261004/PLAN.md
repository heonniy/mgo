# PLAN — one physical BR vs CA stress cell

Date: 2026-10-04
Status: owner-authorized.

## Goal

Measure whether the ~21% frozen-resource peer reduction of the selected CA
stress workload translates into real E2E / TPOT benefit, especially in Env 2.

This is one physical cell, not a new sweep.

## Frozen workload

Use exactly:
`ca_stress_workload_search_20261004/selected_manifests/ShareGPT_R8_B8_c30.json`

- sample_seed = 81
- dp_seed = 86
- R = 8
- local B = 8
- global N = 64
- cache = 30%
- decode = 256
- Gate W128
- substitution = OFF
- BR seed = 42

Do not resample requests, change rank membership, tune the cache ratio, or
change the BR seed after seeing GPU timing.

## Why BR seed 42 is acceptable here

The primary stress workload was selected by the median absolute communication
saving over the fixed BR seeds [7,19,42,73,99,131,181,251], not by seed42 alone.

For this frozen R8/B8/cache30 cell:
- median peer reduction = 20.9679%
- min over the eight BR seeds = 20.7803%
- max = 21.0513%
- best observed BR seed = 42

So seed choice changes the resource reduction by only ~0.27 percentage points
across the audited seeds. Seed42 is used for the physical run because it is the
frozen best observed seed and also the historical default, but report the
eight-seed range next to the physical result.

## Stable measurement harness

Reuse the validated settings from timing_stability_numa_20261004:
- BOUNDARY-only safety monitoring during timed execution;
- no concurrent PSS/smaps walk;
- fixed disjoint CPU affinity per rank;
- NORMAL expert residency behavior (PRETOUCH was not validated as the cause);
- same compile/no-recompile checks;
- same outer-boundary synchronization;
- no profiler or per-event logging.

The 256-step harness previously achieved:
- Env 1 spread 1.03%
- Env 2 spread 0.84%.

## Execution structure

For each policy and environment:
1. PLAN: untimed; compute/freeze BR or CA schedule.
2. COMPILE/WARMUP: untimed.
3. MEASURE: clean frozen-schedule physical execution.
4. COUNTERS: separate untimed pass.

Hungarian optimization is PLAN-only and must not execute in MEASURE.

## Matrix

Exactly four policy/environment combinations:
- Env 1 / BR
- Env 1 / CA
- Env 2 / BR
- Env 2 / CA

Three clean MEASURE repeats each = 12 timed generations.

Counterbalance order:
- repeat 1: Env1 BR -> Env1 CA -> Env2 CA -> Env2 BR
- repeat 2: Env2 BR -> Env2 CA -> Env1 CA -> Env1 BR
- repeat 3: Env1 CA -> Env1 BR -> Env2 BR -> Env2 CA

If either policy within one environment exceeds the validated 5% spread, run
exactly two additional repeats for that environment/pair only, then stop.

## Required metrics

Timing:
- E2E median + full range
- TPOT median + full range
- decode wall median + full range
- BR->CA relative gain for Env 1 and Env 2

Separate COUNTERS:
- peer bytes
- H2D bytes / fetch count
- exact global hit
- exact local hit
- evictions / reloads
- max-min mandatory miss quota per event

Validation:
- output token hash
- frozen route/action hash
- final cache-state hash
- no compilation in MEASURE
- physical cache capacity
- real H2D > 0
- transport preflight for Env1 P2P/IPC and Env2 P2P-disabled SHM

## Interpretation

Primary questions:
1. Does the ~21% peer-byte reduction reduce E2E/TPOT?
2. Is the timing benefit larger in Env 2 than Env 1?

Do not call this dataset-average behavior. Label it:
"optimized communication-stress workload".

Do not run a neutral control, second batch/cache/rank point, substitution,
CA-rep, or LRU automatically after this packet.

Commit results and stop for owner review.
