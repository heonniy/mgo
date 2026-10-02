# AGENT TASK — Exact rank-demand oracle and GPU critical-path validation

Read:

1. README.md
2. PLAN.md
3. matrix.json
4. ../admission_trajectory_controller_breakdown_20261002/RESULTS.md
5. ../../AGENTS.md

## Goal

Test whether the communication-aware policy's expert-work skew is a real TPOT/E2E bottleneck.

Implement an exact diagnostic admission oracle that **ignores communication** and minimizes the busiest rank's current effective expert-token rows under the existing hard admission-count quota.

## Hardware

Use only physical GPUs 0,1,4,5 as R4.

No R8 launch.

## Oracle

After substitution, compute exact current-event demand for every execution expert.

Existing resident owners are fixed.

For incoming exact misses solve the min-max load MILP from PLAN.md.

Use exact SciPy/HiGHS MILP and deterministic optimal tie-break.

No heuristic fallback.

First run an oracle planning pass and save every assignment/cache/token hash.

For timing, inject the saved assignments into a fresh run. The timed replay must reproduce the planning trajectory exactly. Oracle solve time is reported separately and excluded from E2E, so O0 is explicitly an upper-bound diagnostic.

## Timing

Run P0 Random, P1 Hungarian-current and O0 frozen oracle at:

- R4/B4
- R4/B8
- R4/B16

64 decode forwards.

Six uninstrumented repetitions. Use every permutation of the three policy orders once per batch.

Do not use profiler timing for the primary TPOT/E2E result.

## GPU critical path

After uninstrumented timing, profile R4/B8 once for each policy.

Add diagnostic-only phase ranges for:

- routing metadata collective;
- H2D fetch;
- dispatch;
- expert execution;
- combine;
- complete MoE layer.

Extract per-layer/per-step max-rank expert GPU time and compare it with max-rank expert-token rows.

## Required analysis

Answer:

1. Does P1's higher busiest-rank expert-row count become higher expert GPU time?
2. Does O0 reduce max-rank expert GPU time?
3. Does that reduction lower TPOT/E2E?
4. What communication penalty does O0 pay?
5. Are H2D/fetch/controller changes large enough to confound the interpretation?

## Outputs

Publish:

- IMPLEMENTATION.md
- measurement_manifest.json
- oracle_solver_validation.json
- oracle_planning_summary.csv/json
- e2e_repeats.csv/json
- e2e_summary.csv/json
- rank_load_summary.csv/json
- gpu_phase_profile.csv/json
- RESULTS.md
- validation/provenance/hash receipts.

Do not commit giant Nsight traces.

## Stop

Stop after 54 uninstrumented generations, three B8 profiles and the result packet.

Do not design the joint communication+load method yet.
