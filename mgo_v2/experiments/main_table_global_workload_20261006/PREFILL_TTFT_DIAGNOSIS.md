# Prefill TTFT gap: current diagnosis and next diagnostic

Date: 2026-10-07
Scope: R4/C30 main-table, Qwen3-30B-A3B-Instruct-2507, GPUs 0/1/4/5.
This note is diagnostic only. It does not change the selected production runtime.

## Question

Why does OURS have a much larger B64/L512 TTFT than DeepSpeed ZeRO-Inference
although both systems use the same global workload (256 requests), the same
rank-local batch (64 requests/rank), and a <= C30 live GPU-parameter/expert
residency bound?

The earlier explanation "DeepSpeed fetches one expert once and amortizes it
across many prefill tokens" is not a differentiator. OURS also fetches each
miss expert once to its selected owner rank and all routed tokens can reuse that
resident copy.

## What is actually different

OURS performs a global routing/placement protocol on every MoE layer:

1. compute local router outputs;
2. collect cross-rank routing metadata;
3. copy controller metadata to CPU/NumPy;
4. run a replicated global placement decision;
5. dispatch token activations to the chosen expert-owner ranks;
6. wait for required expert H2D under the current prefill barrier;
7. execute experts;
8. return rank partials / activations.

DeepSpeed ZeRO-Inference keeps each rank's token batch local. It materializes
the parameters needed by that rank and does not run OURS' global expert-owner
placement, token dispatch A2A, or return A2A protocol.

Therefore the relevant trade-off is not "amortized fetch versus unamortized
fetch". It is closer to:

- OURS: less expert replication, but global metadata/controller work plus token
  movement across ranks.
- DeepSpeed: parameter streaming/materialization on each rank, but rank-local
  token execution and no expert-owner token A2A.

## Current metadata evidence

The original prefill path gathered full 128-way router probabilities for every
prefill token in addition to selected expert IDs and routing weights. At
B64/L512 there are 131,072 global prompt tokens. The analytical full probability
array alone is 64 MiB per layer after global concatenation (131072 x 128 x
FP32), before selected IDs/weights and CPU policy/layout work.

Commit b32822b6 introduced an opt-in exact Gate-history tail optimization:
only the exact global last 128 probability rows needed by the W=128 Gate
history are retained, while selected expert IDs/routing weights remain exact for
all tokens. It also enabled packed/fused prefill transport while deliberately
retaining the existing required-H2D barrier and no-overlap behavior.

Measured follow-up at branch history 02154188:

- B64/L512 historical owner rerun TTFT: 24.240570 s.
- optimized prefill TTFT samples: 21.664470 s, 20.112728 s.
- optimized mean: 20.888599 s.
- sequential observed reduction: 13.83%.
- timing remains unstable (7.43% two-sample spread).

This is evidence that the original metadata/transport path was a meaningful
contributor, but it cannot explain the full gap to the ~8 s DeepSpeed TTFT.
Do not attribute all of the improvement to metadata alone because the same
experiment also changed prefill transport, and it was not interleaved.

## Important remaining costs

The optimized prefill still has the following characteristics:

- selected expert IDs and routing weights are global for every token;
- CPU policy/layout work remains token-level;
- required H2D completion/global barrier remains in prefill;
- H2D/communication/compute overlap remains disabled for prefill;
- expert execution is H0 per-expert execution, not grouped-GEMM fusion;
- token forward and return communication remain part of the EP path.

These are now stronger candidates than the full probability payload alone.

## Required causal diagnostic

Use B64/L512 first. Run a diagnostic-only one-token generation so measured wall
time is effectively prefill-to-first-token. Do not use this diagnostic timing as
a headline number.

Measure the following exclusive or carefully reconciled spans for all 48 MoE
layers and report both rank mean and critical-rank/global completion:

1. router/Gate compute;
2. routing metadata exchange, separated into:
   - counts,
   - selected expert IDs,
   - routing weights,
   - probability/Gate-history payload;
3. device-to-host conversion / CPU materialization for controller inputs;
4. current placement/controller + layout construction;
5. required expert H2D:
   - bytes/copies,
   - copy-stream service,
   - exposed wait before expert execution;
6. forward token dispatch A2A;
7. expert execution;
8. return A2A / rank-partial combine;
9. attention + dense/non-MoE residual;
10. global/rank waiting that is not already charged to the above.

The accounting must distinguish overlapping service from exposed critical-path
wait. In particular, do not add H2D DMA service to an additive TTFT partition
when it overlaps another phase.

## Decision test

The diagnostic should answer these concrete questions:

- Is the remaining ~20 s dominated by H0 expert compute at large prefill?
- Is forward/return A2A a material fraction of TTFT?
- Does the prefill H2D global barrier expose substantial idle time?
- Is CPU token-level layout/controller work still large after probability-tail
  reduction?
- Is non-MoE attention/dense work already a large unavoidable component?
- Which term grows most from B16/L256 to B64/L512?

Only after this breakdown should we decide whether the next production change
is (a) compact histogram/controller metadata, (b) prefill H2D/comm overlap,
(c) grouped expert execution, or (d) no further prefill optimization.

## Fairness reminder

DeepSpeed's B64/L512 run is not receiving a larger C30 allowance:

- C30 target per rank: 4,348,182,528 bytes (~4.05 GiB).
- observed DeepSpeed expert live peak: 2,925,527,040 bytes/rank.
- observed all-parameter live peak: 3,115,216,896 bytes/rank.

Thus the current TTFT gap should not be explained as DeepSpeed simply receiving
more live GPU parameter memory. The systems use different offload/execution
semantics under the same live-residency ceiling.
