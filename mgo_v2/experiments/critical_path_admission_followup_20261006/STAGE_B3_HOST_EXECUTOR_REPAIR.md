# Stage B3 — Host-side expert executor repair

Parent result: `0848cd0` (B2 diagnostic complete)

Status: owner-authorized runtime repair. Stage C oracle remains blocked.

## 1. Why this stage exists

B2 showed that the large NCCL residency is mostly **waiting for ranks to arrive**, not wire transfer:

- forward inflation explained by rank start skew: 95–97%;
- return inflation explained by rank start skew: 99%+.

However, current calibrated expert GPU service explains only about 4% of the BR->FCA arrival-skew delta.

The dominant expert-loop budget is host-side invocation / enqueue / wrapper work. In the B2 diagnostic captures:

- compiled expert GPU work is only ~18 ms/step mean-rank;
- the host expert loop is ~0.70–0.99 s/step;
- compiled-function host invocation/enqueue alone is ~0.44–0.61 s/step.

Therefore the current runtime is not a clean substrate for admission research. Before another placement model or oracle, reduce the per-expert host-launch path **without changing admission/cache/communication semantics** and remeasure BR/FCA.

This is a runtime repair, not a paper method.

## 2. Fixed scientific scope

Primary cell:
- R4 physical GPUs 0,1,4,5;
- C30 / local B128;
- V3_OPT_PF_OVERLAP;
- P=2 / T2;
- BF16;
- substitution OFF;
- same frozen requests/routes/weights/teacher tokens as 9c20847/B2;
- Env1 normal NCCL.

Policies:
- BR
- FCA

C60/B128 is confirmation-only after the C30 repair gate passes.

No LA_CA/OLD_CA, R8, batch sweep, cache sweep or oracle in this stage.

## 3. H0 — current executor baseline

H0 is the exact current `DecodeOffloadRuntime.compute()` path:

```
for expert group:
    readiness selection / wait
    received[rows]
    self.kernel(...)
    record_slot_use
    part * routing_weight
```

Preserve H0 unchanged as a selectable runtime mode and correctness reference.

## 4. H1 — graph-replay expert-kernel executor

The first repair is deliberately narrow: remove the expensive `torch.compile` host invocation for every expert while preserving the exact expert arithmetic.

### 4.1 Graph cache

Add an opt-in executor, e.g. `GraphExpertExecutor`.

Cache CUDA graphs by:

```
(slot_id, exact_row_count)
```

Each graph entry owns:
- persistent BF16 input scratch `[n, 2048]`;
- persistent output scratch `[n, 2048]`;
- a CUDA graph that executes the already-compiled exact Qwen expert function using the **same cache-slot memory pointer**.

The graph reads weights from `self.cache[slot]`. Eviction may change the contents of that slot, but not its device address, so replay must consume the current slot contents.

Do not copy expert weights into graph-private storage.

### 4.2 Execution

For an expert group:

1. preserve the current ready-first slot selection;
2. gather `received[rows]` into the graph entry input scratch;
3. replay the captured graph;
4. preserve `record_slot_use(slot)`;
5. apply the same routing-weight multiplication;
6. return parts in the same logical group positions.

H1 changes only how the compiled expert computation is launched.

It does **not** change:
- expert ownership;
- ordering chosen by ready-first;
- H2D scheduler;
- cache slot/victim trajectory;
- packet layout;
- forward/return collectives;
- BF16 arithmetic;
- combine order.

### 4.3 Pre-capture discipline

No graph capture or first-time compilation may occur in MEASURE.

Before primary timing:

1. run the full frozen schedule in a discovery/warmup pass;
2. collect all required `(slot,row_count)` signatures;
3. synchronize H2D/background work;
4. build/capture every graph entry;
5. rerun one untimed full schedule and require zero new graph entries;
6. enable `error_on_recompile=True` during MEASURE.

If a new graph signature appears during MEASURE, invalidate that run.

### 4.4 Correctness

H1 must reproduce H0 exactly for:
- frozen selected experts/weights;
- policy assignments;
- victims/cache state hashes;
- H2D copy counts/bytes;
- forward/return split counts and bytes;
- generated token hash.

Compare expert outputs on an untimed prefix. Require exact equality when the graph executes identical compiled kernels; if a CUDA-graph scheduling detail changes floating association outside the expert kernel, report max abs/rel difference and require the same final argmax/token hash. Do not silently change accumulation precision.

## 5. Optional H1b — wrapper consolidation

Run only if H1 removes compiled-host-call overhead but the host loop remains dominant.

H1b may move **gather + graph replay + routing-weight multiply** behind a small executor object with preallocated scratch, so per-expert Python tensor/view construction is minimized.

Constraints:
- no grouped/padded expert arithmetic yet;
- no new fused MoE kernel;
- no changed expert order;
- no extra expert compute;
- no semantic changes.

H1b is still a runtime repair.

## 6. What not to do yet

Do not jump directly to a grouped GEMM / Triton MoE kernel in this checkpoint.

That could be the eventual production repair, but it changes enough of the GPU execution path that it would confound the immediate causal question:

> Does removing per-expert host invocation reduce rank arrival skew and make the placement comparison physically interpretable?

First answer that with H1/H1b.

## 7. Measurement protocol

### 7.1 Diagnostic capture

For C30 BR/FCA, compare H0 vs H1 using the B2 instrumentation.

Report:
- host expert-loop span;
- compiled-function host-call span;
- GPU compiled expert service;
- wrapper GPU service;
- forward host-entry spread;
- forward GPU-start spread;
- return host-entry spread;
- return GPU-start spread;
- forward/return NCCL residency;
- H2D ready-wait upper bound.

Do not use instrumented TPOT as primary performance.

### 7.2 Clean physical timing

After correctness and diagnostic gates, run uninstrumented paired timing:

```
BR/H0
FCA/H0
BR/H1
FCA/H1
```

Use interleaved/counterbalanced order and the existing stable repeat rule.

Primary:
- decode TPOT;
- E2E.

Secondary:
- host loop time from separate capture;
- arrival skew;
- NCCL residency;
- H2D bytes/copies;
- packet counts;
- expert GPU time.

## 8. C30 repair gate

H1 is a successful runtime repair only if all hold:

1. correctness parity passes;
2. mean compiled-function host-call budget drops by >=70%;
3. full expert host-loop span drops by >=40%;
4. no material increase in H2D bytes/copies or packet bytes;
5. absolute BR TPOT improves or stays within 1% while host overhead falls;
6. primary timing is stable under the existing repeat rule.

Then inspect FCA.

### Causal interpretation

#### Case A — FCA penalty shrinks materially

If:

```
(FCA/BR slowdown under H1)
 <= 0.5 * (FCA/BR slowdown under H0)
```

and rank arrival skew shrinks correspondingly, the old placement result is strongly contaminated by the host executor.

Then expand H1 to C60 and re-run BR/FCA there.

#### Case B — H1 fixes host overhead but FCA penalty remains

If host-loop reduction passes but FCA/BR slowdown changes by <25%, the FCA penalty is not mainly the per-expert compiled-call artifact.

Then preserve H1 as the cleaner runtime and return to GPU/communication attribution before any oracle.

#### Case C — H1 does not reduce host loop

Then CUDA-graph replay is not the right repair. Stop and profile the remaining host path before implementing grouped GEMM.

## 9. C60 confirmation

Run only after C30 Case A or B establishes a working H1 runtime.

Same matrix:

```
BR/H0, FCA/H0, BR/H1, FCA/H1
```

No retuning.

## 10. Production follow-up gate

Only after H1 demonstrates that host-side launch overhead is a real confounder should a production executor be designed.

Likely production candidates:
- grouped GEMM / grouped MoE kernel;
- C++/CUDA-side expert launch loop;
- event-level CUDA graph only if a deployable dynamic-shape strategy is demonstrated.

Do not choose among them in this checkpoint.

## 11. Required artifacts

Create:
- `B3_IMPLEMENTATION.md`
- `B3_GRAPH_SIGNATURES.json`
- `B3_CORRECTNESS.json`
- `B3_DIAGNOSTIC_RESULTS.csv/json`
- `B3_TIMING_REPEATS.csv/json`
- `B3_RESULTS.md`
- source/provenance/hash receipts.

Do not commit CUDA graph dumps or large profiler traces.

## 12. Stop discipline

After C30, stop if H1 fails the repair gate.

If C30 passes, run the single C60 confirmation matrix and then stop for owner review.

Stage C oracle remains NOT authorized.

Do not:
- change admission objectives;
- retune FCA;
- introduce LA_CA;
- add substitution/replication;
- use a new batch/cache setting;
- run R8;
- claim H1 itself as the research contribution.
