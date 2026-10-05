# AGENT TASK — Critical-path attribution before the next admission policy

Read in order:

1. README.md
2. PLAN.md
3. matrix.json
4. ../decode_prefetch_runtime_refactoring_20261004/POLICY_REGIME_FINAL_RESULTS.md
5. ../../AGENTS.md

## Goal

Explain why FCA reduced remote packet count but increased TPOT, then measure the maximum useful headroom of current-layer admission under a physically calibrated objective.

Do not implement a new production admission heuristic first.

## Stage order

### A1. A2A split-shape microbench
Use R4 GPUs 0,1,4,5 and the same normal NCCL environment as the Env1 policy-regime run.

Hold global packet count fixed while changing directed source/destination split shape.

Measure BF16 forward-sized (4120 B) and return-sized (4096 B) packets.

### A2. Return arrival-skew microbench
Hold split matrix/payload fixed. Delay one rank on the GPU stream before return A2A by calibrated 0/0.10/0.25/0.50/1/2 ms.

Rotate the delayed rank.

### A3. Expert tau(n)
Use the real compiled expert kernel. Measure n=1..512 powers of two and validate additive prediction on real multi-expert row bundles.

STOP and publish MICROBENCH_RESULTS before doing Stage B.

### B. Critical-path model
Use only A1-A3 to calibrate the model. Do not fit coefficients to 9c20847 TPOT.

Validate it against the existing C30/C60 BR/OLD_CA/FCA/LA_CA mechanism captures.

Do not proceed unless it reproduces the policy ordering and passes the PLAN gates.

### C/D. Exact current-layer oracle and frozen replay
Only after Stage B passes, solve the current-layer placement problem under the same balanced H2D quota using the validated physical cost.

Do not read future routes.

Save assignments in a planning pass, then time a frozen replay at C30/B128 and C60/B128.

The expensive solver is not part of timed TPOT.

## Primary decision

Use physical oracle TPOT headroom:

- >=5% one cell and >=3% the other: GO for online CPA.
- 2-5%: marginal; review.
- <=2% both cells: stop current-layer admission work and pivot to multi-step cache/eviction trajectory optimization.

## Non-negotiable fairness

- same R4 GPU set 0,1,4,5;
- V3 P2/T2 for oracle replay;
- BF16;
- substitution OFF;
- same frozen requests/routes/weights/teacher tokens;
- same eviction/cache semantics;
- same mandatory-fetch quota;
- no replication/migration;
- no latency-based sample exclusion.

Do not use instrumented runs as primary TPOT.

Do not terminate foreign jobs.

Do not commit giant profiler files.
