# C++ expert executor validation results

R4 C60 B64 local/input256/64 decode; frozen routing and teacher tokens; prefetch OFF; two primary repeats per backend.

|Backend|TPOT samples s|Mean TPOT s|Relative range %|Mean TTFT s|Mean E2E s|
|---|---|---:|---:|---:|---:|
|h0|0.913145 / 0.893301|0.903223|2.20|2.783684|60.589957|
|native|0.671119 / 0.667127|0.669123|0.60|2.768103|45.591963|

Observed TPOT reduction: 25.92%. All four runs retain exactly the same expert routing, weights/history, teacher inputs, controller counters, final cache state/role hashes, decode H2D bytes and peer bytes. Predicted token agreement is100% against H0. No primary has detailed phase instrumentation. Prefill remains H0.

Run order is H0/native/native/H0. Two measurements per backend show a native relative range of0.60% versus H0 2.20%; this is not evidence of long-term jitter elimination. Full-model generation uses fixed routing/teacher inputs; unrestricted greedy, prefetch ON, other batches/ranks and prefill migration are not validated here. No production default change.

## Implemented scope

The C++ extension executes per-expert input gather, three BF16 matrix multiplications, fused SiLU/multiply and routing-weight application within one native call per ready subset. It releases the Python GIL, accesses existing cache slots directly, and uses the current CUDA stream. Python retains wave readiness and H2D scheduling. Shared slot-use events preserve overwrite ordering. There is no grouped GEMM or legacy slot-to-parameter copy.

In the first native full-model repeat, four ranks executed86900–88868 expert invocations over6022–6106 ready waves. Average wave size was14.43–14.65 experts, with roughly two waves per layer. Group size follows actual readiness rather than waiting to fill a fixed batch.

## Validation

- Four-GPU synthetic BF16 checks at1/3/5/17/65 rows per expert: relative L2 and maximum absolute differences were zero in tested cases. This is observed parity, not a general bitwise guarantee.
- Real pinned H2D not-ready path, non-default compute stream, immediate slot overwrite and output ordering: PASS on all four GPUs.
- Four CPU scheduling/factory tests plus two existing scheduler tests: PASS.
- Synthetic32-expert timings: H0 277–285us/expert, C++ single calls170–174us, C++ waves166–169us (means across ranks/repeats for tested sizes). These are resident microbenchmarks, not TPOT.

## Causal limits and next migration boundary

The improvement is direct evidence that the changed expert execution path matters. It is not a measurement of Python interpreter cost alone: C++ also changes PyTorch dispatch/FFN packaging and batches readiness/event handling. The C++ single-vs-wave comparison shows most synthetic improvement already occurs when the inner expert path is native.

Copying the entire legacy coslot engine is not justified by these results. Its dispatcher includes per-expert weight D2D copies and CPU completion waits absent from this implementation. A further native scheduler should preserve the current cache views and CUDA dependencies. Native queue/slot management could reduce remaining host overhead, but would need additional measurement; hardware/OS/collective variability would remain.

Raw receipts are archived in full_model_v1 and micro_v1; supervisor logs/resources remain under the corresponding /home/hwlee/mgo-results/headline_r4_20261007 directories. Only GPUs0/1/4/5 were used; owned model loads were restored. See README.md for the opt-in --expert-executor native path.
