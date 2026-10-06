# Stage B4 — Dynamic grouped-GEMM expert executor

Parent result: `1fc6278` (B3 C30 H1b timing complete)

Status: owner-authorized runtime implementation/validation plan. Stage C oracle remains blocked.

## 1. Motivation

B3 showed that H1b removes a large fraction of the per-expert launch artifact:

- BR TPOT: 1.826 -> 1.468 s (-19.6%);
- FCA TPOT: 1.979 -> 1.522 s (-23.1%);
- FCA-vs-BR slowdown: 8.41% -> 3.68% (-56.3%);
- compiled host-call budget: -85%+.

However H1b still executes one expert at a time from Python and requires a frozen-signature CUDA-graph cache:
- 38k-42k graph entries/rank;
- roughly 30-38 GiB persistent scratch/rank;
- graph capture depends on discovering the frozen schedule beforehand.

H1b is therefore a useful diagnostic substrate, but it still couples expert placement to:
- per-expert gather calls;
- per-expert graph replay calls;
- per-expert routing-weight operations;
- per-expert slot-use bookkeeping;
- per-expert ready polling.

B4 replaces that loop with a dynamic grouped expert executor so the measured runtime is closer to the intended physical knobs: H2D readiness, grouped expert work, and inter-rank communication.

This is still runtime repair, not the admission contribution.

## 2. Research question

Does removing per-expert host orchestration reduce the gap between the **placement-visible work** and the **measured rank completion time**?

The desired path is:

```
expert->rank assignment
 -> H2D readiness
 -> grouped expert GPU work
 -> rank-ready time
 -> collective arrival skew
 -> TPOT
```

rather than:

```
expert->rank assignment
 -> number/order of Python calls
 -> host progress skew
 -> collective arrival skew
 -> TPOT
```

B4 must quantify that change; a faster TPOT alone is not sufficient.

## 3. Fixed scope

Use only physical GPUs 0,1,4,5. Do not launch on or alter GPUs 2,3,6,7.

Primary:
- R4;
- C30;
- local B128;
- decode64 for clean timing;
- V3_OPT_PF_OVERLAP;
- P=2 / T2;
- BF16;
- substitution OFF;
- Env1 normal NCCL;
- frozen requests/routes/weights/teacher tokens;
- policies BR and FCA.

C60/B128 is confirmation-only after the C30 H2 gate passes.

H0 and H1b remain available as references. Do not delete or silently replace them.

## 4. H2 executor — ready-wave grouped GEMM

### 4.1 No frozen graph signatures

H2 must be dynamic.

It may allocate bounded reusable workspace sized by the maximum event rows, but it must not:
- inspect future events;
- pre-capture one object per (slot,row-count) signature;
- allocate memory proportional to the number of frozen signatures.

Report persistent and peak workspace explicitly.

### 4.2 Ready waves

Preserve H2D/compute overlap.

At an event, maintain the original expert-group list and repeatedly:

1. batch-query which pending expert slots are physically ready;
2. form one wave from **all currently ready** groups;
3. if no group is ready, wait for the next pending demand slot using the existing scheduler semantics, then re-query;
4. execute the whole ready wave with grouped kernels;
5. write each result back to its original group index.

Do not insert a global "all experts ready" barrier.

Record:
- number of experts per wave;
- waves per event/rank;
- rows per expert in each wave;
- time spent waiting for the first/next wave.

### 4.3 Batched scheduler operations

Add decision-equivalent scheduler helpers for H2 so orchestration itself is not per-expert Python work:

- `ready_many(slots)`: inspect a list under one scheduler critical section and return a readiness mask;
- `wait_for_slot(slot)`: retain current semantics when no wave is ready;
- `record_slots_use(slots)`: record one completion event after a grouped wave and associate that event with every cache slot consumed by the wave.

H0/H1b semantics must remain unchanged.

### 4.4 Packed wave layout

For each ready wave build compact metadata:

```
slot_ids[k]
row_offsets[k+1]
row_counts[k]
packed_row_indices[total_rows]
packed_routing_weights[total_rows]
original_group_ids[k]
```

Gather all wave inputs with one packed gather/index-select into a reusable BF16 input buffer.

The output must be addressable by the original group order so the existing rank-partial accumulation/combine order is preserved.

## 5. Grouped expert math

Qwen expert:

```
gate = X @ W_gate^T
up   = X @ W_up^T
act  = silu(gate) * up
Y    = act @ W_down^T
```

All experts share shapes; only row counts and cache slots differ.

### 5.1 Preferred H2 kernel structure

Implement dynamic grouped GEMM over the live cache arena:

1. grouped gate+up projection for all experts in a ready wave;
2. fused SiLU * up;
3. grouped down projection;
4. vectorized routing-weight multiply.

The kernel receives live `slot_ids` and computes weight addresses from `self.cache`; it must not own a private copy of expert weights.

A combined gate+up view is allowed because the cache stores gate and up contiguously, but report any numerical difference from H0/H1b.

### 5.2 Backend order

Preferred backend:
1. a dynamic Triton grouped-GEMM kernel compatible with the current H100/BF16 shapes;
2. a supported PyTorch/CUDA grouped-GEMM primitive if it provides the same dynamic semantics.

Do not use a private/unsupported API without a committed capability receipt and fallback.

If neither path can execute the exact variable-row workload reliably, stop and report rather than padding every expert to a large fixed M.

## 6. Correctness gates

Before timing, H2 must preserve exactly:
- routing inputs;
- admission assignments;
- cache/victim trajectory;
- H2D copies and bytes;
- forward/return split counts and bytes;
- ready-first causal legality;
- generated token hash;
- final cache/controller hashes.

Grouped GEMM may use a different GEMM kernel than H0/H1b, so expert tensors need not be bitwise identical.

On an untimed full decode64 pass report:
- max absolute expert-output difference;
- max relative difference;
- final token/argmax parity.

No change in accumulation precision or return-combine order is allowed.

## 7. H2 micro-calibration

Before policy timing, calibrate only the H2 executor itself.

Use representative real wave row vectors sampled structurally from the frozen C30 trace, plus synthetic balanced/unbalanced vectors with the same total rows.

Measure:

```
T_group(rows_1, ..., rows_k)
```

for:
- 1, 2, 4, 8, 16, 32 experts/wave when present;
- representative total-row levels;
- skewed vs balanced row partitions at matched total rows.

Do not fit coefficients to BR/FCA TPOT.

The output is a small grouped-service LUT/model used only to test whether rank expert completion becomes predictable.

## 8. Diagnostic comparison

Run separate first-8-step diagnostic captures for:

```
BR / H1b
FCA / H1b
BR / H2
FCA / H2
```

H1b can reuse the B3 capture only if runtime/input/system fingerprints match; otherwise recapture it.

Report per rank/event:
- expert groups;
- grouped-wave count and sizes;
- host expert-loop span;
- expert-related host launch/call count;
- grouped GPU active time;
- H2D ready wait;
- forward collective entry/start skew;
- return collective entry/start skew;
- forward/return NCCL residency;
- packet/H2D parity.

### Knob-to-runtime gap metrics

For each event/rank compute:

```
pred_group_service = H2 grouped-service model(row vector)
observed_expert_stage
observed_return_entry_time
```

Report:
- Spearman(pred_group_service, observed_expert_stage);
- aggregate grouped-service relative error;
- BR->FCA predicted expert-stage delta error;
- correlation between predicted rank-ready skew and measured return-entry skew.

Also report raw expert-row count as a baseline.

The point is to determine whether H2 makes a placement-visible quantity more predictive than H0/H1b.

## 9. Clean C30 timing

After correctness and diagnostic checks, run counterbalanced uninstrumented timing:

```
BR / H1b
FCA / H1b
BR / H2
FCA / H2
```

Two repeats initially; use the existing stability rule.

Primary:
- TPOT;
- E2E.

Secondary counters are collected separately.

Do not mix profiler runs with primary timing.

## 10. C30 H2 gate

H2 is accepted as the new evaluation substrate if:

1. correctness/counter parity passes;
2. no frozen-signature graph cache is required;
3. expert-related host launch/call count is reduced by >=80% relative to H1b;
4. expert host-loop span is reduced by >=25% relative to H1b;
5. BR TPOT is no worse than H1b by >1%;
6. clean timing is stable;
7. H2 grouped-service prediction has event-level Spearman >=0.80 with measured expert-stage time;
8. BR->FCA expert-stage delta error is <=25%, **or** the remaining error is shown to come from separately measured H2D readiness rather than host orchestration.

Criterion 8 is the main "knob vs operation" gate.

Do not require FCA to become faster than BR; FCA is a stress policy, not the target method.

## 11. Interpretation

### Case A — H2 closes the gap

If H2 removes most host orchestration and grouped-service + H2D readiness explain rank completion:

```
placement-visible work -> physical rank completion
```

is sufficiently calibrated.

Then run C60 confirmation and return for owner review before any oracle.

### Case B — H2 is faster but placement-visible cost is still poorly predictive

The remaining confounder is not per-expert Python launch.

Inspect H2D staging/readiness and packet preparation next. Do not build an oracle.

### Case C — H2 gives little improvement over H1b

H1b is already adequate as the paper evaluation substrate. Keep H1b and avoid further executor engineering unless needed for reviewer robustness.

## 12. C60 confirmation

Only after C30 Case A:

- C60/B128;
- BR and FCA;
- H1b vs H2;
- same correctness, diagnostic, and clean timing rules;
- no retuning.

The purpose is to check whether the reduced knob/runtime gap survives when H2D pressure is lower.

## 13. Required artifacts

Create:
- `B4_IMPLEMENTATION.md`
- `B4_BACKEND_CAPABILITY.json`
- `B4_CORRECTNESS.json`
- `B4_GROUPED_SERVICE_CALIBRATION.csv/json`
- `B4_WAVE_STATS.csv/json`
- `B4_DIAGNOSTIC_RESULTS.csv/json`
- `B4_KNOB_RUNTIME_GAP.json`
- `B4_TIMING_REPEATS.csv/json`
- `B4_RESULTS.md`
- provenance/hash/resource receipts.

Do not commit large Nsight traces.

## 14. Stop discipline

After the C30 H2 gate:
- stop immediately on Case B/C for owner review;
- run only the single C60 confirmation on Case A;
- then stop for owner review.

Stage C oracle remains NOT authorized.

Do not:
- change admission policy/objective;
- retune FCA;
- add LA_CA;
- add substitution/replication;
- run R8;
- change batch size;
- use future route information;
- claim grouped GEMM as the paper contribution.
