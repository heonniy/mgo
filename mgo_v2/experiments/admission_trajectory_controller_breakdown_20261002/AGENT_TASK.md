# AGENT TASK — Admission trajectory/controller breakdown

Read in order:

1. README.md
2. PLAN.md
3. matrix.json
4. ../local_remote_e2e_impact_20261001/RESULTS.md
5. ../../AGENTS.md

## Objective

Explain, using only new four-GPU runs, how Hungarian-current changes controller/cache trajectory across local batch sizes and why R4/B8 is slower than Balanced Random. Existing R8 results are retrospective context only.

Do not design a new policy yet.

## Required implementation

Add diagnostic-only controller instrumentation, disabled by default.

Break `GlobalExpertController.plan_layer()` into measured subregions, including:

- gate-history update;
- substitution;
- effective-route merge;
- admission preparation/cost-build/assignment;
- eviction candidate/coverage/victim work;
- cache mutation/bookkeeping.

Record operation counts with the timings.

For Hungarian, separate cost construction from `linear_sum_assignment`.

For Coverage, count/split `_sync_coverage` and victim-selection work.

Add tests proving instrumentation on/off preserves decisions and outputs.

## Trajectory data

Per event record:

- misses/admissions/evictions/reloads;
- hit/subhit/miss;
- H2D fetches;
- remote/local pairs;
- rank token load/CV;
- candidate counts;
- cache owner changes.

Track each admission to its next raw demand and report next-use survival/reload behavior.

## GPU scope

Run only on physical GPUs **0,1,4,5**:

- R4/B4 Random + Hungarian-current;
- R4/B8 Random + Hungarian-current;
- R4/B16 Random + Hungarian-current.

Use 64 decode steps, one detailed diagnostic run per condition.

**Do not launch any R8 job.**

These diagnostic wall times are not new speedup claims; use the existing five-repeat E2E study for performance.

## Frozen-route replay

Save raw global router metadata from each diagnostic run to server-side trace files.

Run CPU replays:

1. each policy on its own trace;
2. Random and Current on the same Random trace;
3. Random and Current on the same Current trace.

Each replay keeps its own cache/substitution/eviction state.

Run at least five timing repetitions.

## Analysis

Produce cumulative event-aligned deltas for:

- remote pairs;
- H2D/fetch;
- reload;
- eviction count;
- coverage-sync time;
- total controller time.

Find where positive and negative trajectories diverge.

Retain concrete cache-state examples when possible.

## Required outputs

Publish:

- IMPLEMENTATION.md
- measurement_manifest.json
- controller_breakdown.csv/json
- trajectory_summary.csv/json
- next_use_survival.csv/json
- matched_replay_summary.csv/json
- state_examples.md
- implementation_audit.md
- RESULTS.md
- validation/provenance/hash receipts.

Do not commit giant raw traces.

## Stop

Stop after the bounded diagnostic and replay packet.

Do not optimize the controller, add load constraints, retune policies or add new admission methods before owner review.
