# Stage B2 — Runtime gap attribution

Parent result: `e656666` (Stage B FAIL)

Status: owner-authorized diagnostic follow-up. Stage C oracle remains blocked.

## 1. Why B2 exists

Stage B learned two different things:

1. the calibrated expert service model is useful:
   - eventwise expert-time correlation is 0.966–0.999;
   - aggregate expert-time error is about 23–24%.

2. the full-runtime communication timing is not explained by isolated service microbenchmarks:
   - forward aggregate error is about 98%;
   - return residency aggregate error is about 95%;
   - active forward/expert/return union still has 90–93% aggregate error.

At the same time, the model predicts FCA/BR return inflation well:
- C30 predicted 2.725x vs observed 2.633x;
- C60 predicted 2.374x vs observed 2.417x.

Therefore B2 does **not** try another placement objective. It asks where the missing runtime time comes from and whether that missing time is controllable by expert admission.

## 2. Primary questions

For one real MoE event, separate:

```
CPU/controller
 -> packet preparation
 -> collective enqueue
 -> collective GPU start
 -> collective completion
 -> expert slot readiness
 -> expert wrapper
 -> compiled expert kernel
 -> partial preparation
 -> return enqueue
 -> return GPU start
 -> return completion
 -> combine
```

Answer:

1. Why does an isolated ~0.03 ms A2A become ~1–5 ms rank-local NCCL residency in the real runtime?
2. How much of the expert-loop gap beyond sum tau(n) comes from H2D slot waiting, gather/scatter, weighting and Python scheduling?
3. Is the large return residency mainly late-rank arrival, and what makes the late rank late?
4. Which missing terms are functions of the admission decision, versus common runtime overhead that admission cannot optimize?

## 3. Scope

Use exactly:
- R4 physical GPUs 0,1,4,5;
- local B128;
- V3_OPT_PF_OVERLAP;
- P=2, trigger T2;
- BF16;
- substitution OFF;
- same frozen request/route/weight/teacher inputs as `9c20847`;
- Env1 normal NCCL path.

Primary diagnostic cells:
- C30 / BR
- C30 / FCA
- C60 / BR
- C60 / FCA

LA_CA is a confirmation cell only if the BR/FCA decomposition yields a clear mechanism that needs a balanced-compute control.

OLD_CA is not required in B2.

Use the first 8 decode steps for fine-grained instrumentation unless a required tail event lies outside that prefix. If a tail event is needed, select it by a **predeclared structural criterion** from the existing event table (for example top 1% predicted ready skew), not by observed latency.

B2 is diagnostic. Instrumented TPOT is not a performance claim.

## 4. Instrumentation design

### 4.1 Shared timestamp rule

Do not compare raw CUDA-event timestamps across GPUs.

Use:
- process host `CLOCK_MONOTONIC_RAW` / equivalent monotonic timestamps for cross-rank host entry/exit spreads;
- Nsight/CUPTI global timestamps for cross-GPU kernel-start/kernel-end alignment;
- CUDA events only for rank-local GPU interval duration.

Every event record must include rank, layer, decode step and a deterministic event id.

### 4.2 Forward path ranges

Split the current forward path into:

1. `current_controller_cpu`
2. `demand_h2d_enqueue_cpu`
3. `forward_pack_gpu`
   - hidden gather/copy into payload
   - routing-weight/id packing
4. `forward_collective_host_call`
5. `forward_nccl_gpu`
6. `forward_unpack_gpu`

Record:
- host collective-entry timestamp per rank;
- host entry spread across ranks;
- GPU NCCL kernel-start timestamp per rank from CUPTI;
- GPU start spread across ranks;
- rank-local NCCL kernel residency;
- A1 wire/base prediction from the exact event split matrix.

Derived quantities:

```
forward_host_arrival_spread
forward_gpu_start_spread
forward_wire_hat
forward_residual
```

The purpose is to distinguish payload cost from late collective launch / stream queueing.

### 4.3 Expert path ranges

For every rank/event, measure:

1. `expert_ready_wait`
   - time with no executable expert because required slots are not ready;
2. `expert_gather`
3. `expert_compiled_kernel`
4. `expert_weight_partial`
5. `expert_loop_other`

Keep A3 prediction:

```
expert_tau_hat[r] = sum_e tau(n_e)
```

Compare it with:
- sum of actual compiled-expert spans;
- full rank-local expert loop;
- ready-wait excluded expert loop.

This tells us whether the remaining ~23% expert error is kernel calibration, wrapper overhead, or H2D readiness.

### 4.4 Return path ranges

Split return into:

1. `return_partial_build_gpu`
   - zero/init partial buffer;
   - index_add / rank-partial accumulation;
2. `return_collective_host_call`
3. `return_nccl_gpu`
4. `return_output_combine_gpu`

Record:
- last required expert/partial-ready time per rank;
- host return-collective entry per rank;
- CUPTI NCCL kernel-start per rank;
- rank arrival spread;
- rank-local NCCL residency;
- A1 return wire/base prediction.

Derived:

```
return_ready_spread
return_host_arrival_spread
return_gpu_start_spread
return_wire_hat
return_wait_hat_from_A2
return_residual
```

A2 predicts approximately one-for-one propagation from late arrival to collective completion. B2 tests whether the *actual* full-model return tail follows this relation.

## 5. Attribution accounting

For forward and return, decompose the observed rank-local collective interval into the following conceptual terms without double counting:

```
observed_collective_tail
  ~= base_wire_service
   + arrival/launch skew
   + unexplained residual
```

Do not call the entire NCCL kernel residency "wire latency".

For the expert path:

```
observed_expert_loop
  ~= sum tau(n_e)
   + slot_ready_wait
   + gather/weight/partial overhead
   + unexplained residual
```

The accounting is diagnostic, not an additive TPOT decomposition.

## 6. Placement-causal classification

Each discovered term must be labeled as one of:

### A. Admission-controllable
Examples:
- expert service imbalance caused by owner assignment;
- return arrival skew caused by owner-dependent expert completion;
- split-matrix concentration caused by owner assignment;
- owner-dependent H2D slot waits.

### B. Policy/runtime-dependent but not directly placement-cost-predictable
Examples:
- policy-specific host controller skew;
- variable Python scheduling or allocator behavior.

### C. Common/non-controllable
Examples:
- fixed NCCL launch tax;
- common packet pack/unpack overhead;
- fixed combine cost.

This classification matters more than explaining 100% of absolute TPOT. A current-layer oracle is only justified by terms in class A that can be predicted before executing the layer.

## 7. B2 validation gates

B2 does not reuse the failed Stage-B absolute model blindly.

### Mechanism gate

For BR and FCA in both C30/C60:

1. explain at least 80% of the **forward NCCL residency inflation over A1 wire/base** using measured launch/arrival/queue terms, OR show that the residual is policy-independent within 10%;
2. explain at least 80% of the **return NCCL residency inflation over A1 wire/base** using measured rank-ready/arrival skew plus fixed runtime terms, OR show that the residual is policy-independent within 10%;
3. explain the expert-loop aggregate to <=25% relative error after adding measured wrapper/ready-wait terms.

### Placement-causality gate

Compute the BR->FCA change in each attributed term.

PASS only if the dominant BR->FCA slowdown can be assigned to one or more admission-controllable class-A terms.

If the dominant slowdown is class B or unexplained, the current-layer oracle remains blocked.

If the dominant missing time is class C and approximately common across policies, it may be removed as a constant when building a **delta cost model**, but this must be demonstrated rather than assumed.

## 8. Revised-model option after B2

Only if B2 passes, build a revised model from admission-controllable terms:

```
Delta T_hat(x)
 = Delta forward_controllable(x)
 + Delta expert_ready(x)
 + Delta return_wait_controllable(x)
 + Delta H2D_wait_controllable(x)
```

Do not fit an arbitrary scale to old TPOT.

Validate the revised model on the held-out C30/C60 BR/OLD_CA/FCA/LA_CA events.

Required before any oracle:
- policy ordering reproduced in both caches;
- event-level Spearman >=0.60 for controllable-delta target;
- aggregate BR->policy delta error <=25%.

Absolute common runtime offset is no longer a gate if B2 proves it is policy-independent.

## 9. Outputs

Create:
- `B2_INSTRUMENTATION_MANIFEST.json`
- `B2_EVENT_TIMELINE.csv/json`
- `B2_FORWARD_ATTRIBUTION.csv/json`
- `B2_EXPERT_ATTRIBUTION.csv/json`
- `B2_RETURN_ATTRIBUTION.csv/json`
- `B2_PLACEMENT_CAUSALITY.json`
- `B2_RESULTS.md`
- source/provenance/hash receipts.

Do not commit large Nsight reports.

## 10. Stop discipline

After B2 attribution and optional revised-model validation, STOP for owner review.

Stage C oracle is still NOT authorized.

Do not:
- implement CPA;
- retune FCA/LA_CA;
- change runtime/cache/batch;
- run R8;
- use observed latency to cherry-pick events;
- fit a free lambda or global scaling coefficient to make old TPOT match.
