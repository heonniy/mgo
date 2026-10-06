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

Checkpoint A is complete at `dd28e4d`. Stage B is now owner-authorized. Read `STAGE_B_EXECUTION.md` before starting Stage B.

### B. Critical-path model
Stage B is authorized.

Use only A1-A3 to calibrate the model. Do not fit coefficients to 9c20847 TPOT.

Validate it against the existing C30/C60 BR/OLD_CA/FCA/LA_CA mechanism captures.

Use the no-double-counting return model in `STAGE_B_EXECUTION.md`:
- expert skew determines relative rank-ready times;
- return residency may include waiting for the last rank;
- do not independently add the same waiting twice.

Publish Stage-B validation and STOP for owner review even if it passes.

### B2. Runtime gap attribution
Stage B failed at `e656666` because isolated A2A/expert service times under-predict full-runtime forward/return active time by 90%+.

B2 is now owner-authorized. Read `STAGE_B2_RUNTIME_GAP_ATTRIBUTION.md`.

Instrument only diagnostic runs needed to separate:
- forward pack / launch / queue / NCCL;
- expert ready-wait / wrapper / compiled kernel;
- return partial-build / arrival / NCCL / combine.

Primary cells are C30/C60 × BR/FCA at B128. LA_CA is confirmation-only if needed.

After B2 and any revised delta-model validation, STOP for owner review.

### B3. Host-side expert executor repair
B2 completed at `0848cd0` and showed that NCCL residency is dominated by rank-arrival skew, while the current per-expert host invocation path dominates the expert-loop budget.

B3 is now owner-authorized. Read `STAGE_B3_HOST_EXECUTOR_REPAIR.md`.

Implement H1 as an opt-in CUDA-graph replay executor for the existing compiled expert kernel, keyed by cache slot and exact row count. Preserve H0 unchanged.

Primary first cell: C30/B128 with BR and FCA only.

Required order:
1. implementation + full frozen-schedule graph-signature discovery;
2. correctness parity;
3. separate B2-style diagnostic capture H0 vs H1;
4. clean uninstrumented paired timing H0 vs H1;
5. evaluate the C30 repair gate;
6. run C60 confirmation only if C30 establishes a working H1 runtime.

Do not introduce grouped GEMM/Triton yet unless H1 fails and the owner explicitly authorizes another repair.

### B4. Dynamic grouped-GEMM expert executor
B3 completed at `1fc6278`. H1b improved TPOT by 19.6% (BR) / 23.1% (FCA) and reduced the FCA-vs-BR slowdown from 8.41% to 3.68%, but it still executes per-expert Python orchestration and requires 38k-42k frozen CUDA-graph signatures per rank.

B4 is now owner-authorized. Read `STAGE_B4_GROUPED_GEMM_EXECUTOR.md`.

Implement a dynamic ready-wave grouped-GEMM executor (H2):
- no future-event graph discovery;
- no per-signature CUDA-graph cache;
- live cache-slot weights;
- preserve H2D overlap through ready waves;
- batch readiness and slot-use bookkeeping;
- preserve packet/cache/admission semantics.

Primary first cell: C30/B128 with BR/FCA. Compare H1b vs H2 with separate diagnostics and clean timing.

The key acceptance test is not only TPOT. Quantify whether grouped service + separately measured H2D readiness predict measured rank completion substantially better than the old per-expert host path.

Run C60 only if the C30 H2 gate passes.

### C/D. Exact current-layer oracle and frozen replay
NOT authorized in this checkpoint. Do not run any oracle planning or frozen-oracle GPU replay after B4 without owner approval.

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

Do not use instrumented runs as primary TPOT. B2/B3 instrumentation is mechanism-only.

Do not terminate foreign jobs.

Do not commit giant profiler files.
