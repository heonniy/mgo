# B4 dynamic grouped-GEMM executor validation

**H2_NOT_ACCEPTED. No valid primary TPOT/E2E comparison or gain is available.**

R4 physical GPUs 0,1,4,5; C30/local B128; BF16; frozen decode64; Env1; V3 P2/T2. No C60 or Stage C oracle was run.

## Completed implementation and correctness

- Dynamic Triton ready-wave GEMM reads live cache weights, with no future signatures or graph cache. H0/H1b remain available and H2 remains opt-in.
- Standalone persistent workspace: 66 MiB/GPU plus small current-wave metadata.
- BR/FCA full64 checked passes: 648,450 expert comparisons, maximum absolute/relative difference 0, identical tokens and counters in those passes.
- Two CPU scheduler tests pass. Fifty-three isolated service calibration vectors and all four separate first8 captures completed.

## Why timing stopped

The first clean preparation had one canceled 9-MiB BR/H1b prefetch on rank1. Its token hash still matched. Failed H2 rank1 values were not saved before its assertion; their exact outcome cannot be reconstructed. This attempt produced no primary timing samples.

One bounded retry added canonical B3 workload checks and receipt-before-assert reporting. All 16 warm rank checks passed with zero canceled copies. BR/H1b and FCA/H1b each then produced one valid sample. During BR/H2 r1, rank3 (physical GPU5) canceled one prefetch: **55,022 copies instead of 55,023**, or **9 MiB fewer H2D bytes**. All four H2 token hashes and no-compilation checks passed, and controller counters matched. The strict physical-workload comparison failed.

This is timing-dependent cancellation in the existing asynchronous prefetch scheduler, observed in both H1b and H2. It is not evidence of a grouped-GEMM arithmetic error. No forced copy, counter tolerance, or latency-based exclusion was introduced. No further retry was run.

The two completed H1b samples and three partial H2 rank receipts are preserved in B4_TIMING_REPEATS.json and local raw artifacts. They do not meet the two-repeat paired requirement, so no performance gain is reported.

## Separate mechanism evidence

|Policy|Host dispatch reduction|Host-loop reduction|Placement-row/service Spearman|Singleton waves|
|---|---:|---:|---:|---:|
|BR|14.46%|12.40%|0.260|91.19%|
|FCA|15.27%|-16.12%|0.229|91.19%|

Targets were >=80% host dispatch reduction, >=25% host-loop reduction, and Spearman >=0.80. These mechanism gates failed independently of the timing interruption. Host-loop values are single separate instrumented captures, not primary TPOT; do not interpret their differences as stable physical speedups.

Observed-wave service sums correlate strongly with measured duration, but those wave boundaries are physical readiness outcomes. They are retained as post-hoc diagnostics and cannot pass the placement-visible prediction gate. H2D readiness fragments the immediate ready waves into mostly singleton work.

## Resource and provenance audit

Receipt audit passed for this bounded failure report. Minimum host available at timing boundaries: 1659.2 GiB. Maximum GPU allocation recorded by complete samples: 51.16 GiB. No OOM was observed. Only GPUs 0,1,4,5 were used; 2,3,6,7 remain reserved for other users.

All raw attempts, checked outputs, single timing samples, invalid-rank receipts, calibration vectors and diagnostic summaries are retained. No large Nsight file is committed. See B4_AUDIT.json, B4_FAST_PATH_COUNTER_FAILURE.json and B4_REPAIR_DECISION.json. Stop for owner review; no further GPU study is queued.

Owned resident model workers were restored and PID-to-GPU mapping verified on 0,1,4,5 only. See B4_RESOURCE_RESTORATION.json.
