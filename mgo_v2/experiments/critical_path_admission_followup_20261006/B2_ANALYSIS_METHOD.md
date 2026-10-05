# B2 analysis conventions frozen before export

Use the four recovered C30/C60 BR/FCA prefix captures under the observed shared
host state. The original pre-arrival C30/BR is supplemental only. The interrupted
C30/FCA remains a failure receipt and is never a valid sample. Do not stop foreign
GPU2/3/6/7 jobs. Preserve their PIDs in resource snapshots; shared-host load is
a limitation, not proven stationary background.

Map per-rank Nsight timestamps into host CLOCK_MONOTONIC_RAW with matched
NVTX/host-span midpoint anchors. Report slope, residual p50/p99/max; reject
alignment if p99 exceeds50us or slope leaves[.999,1.001]. CUDA-event clocks are
never compared between GPUs.

For each collective/rank/event, measured-start waiting is the last rank GPU
NCCL start minus this rank start. Subtract this and the frozen A1 base exactly
once from residency. Preserve signed residual; gate uses sum(abs(residual))
over sum(max(residency-base,0)), avoiding cancellation. Host arrival, partial
ready spread and enqueue-to-GPU intervals are separate alternative explanations
and are not added again. Residual invariance uses BR/FCA mean absolute residual
relative difference against their pair mean.

Expert host spans form an exact disjoint budget: ready selection minus nested
ready wait; ready wait; gather; compiled host call; record-use; weighting; other.
GPU-active budget instead uses actual compiled GPU intervals plus gather/weight
GPU intervals. Never add host and GPU budgets. Compare tau plus measured GPU
wrapper against GPU active work; tau plus measured noncompiled host budget
against host loop. Host compiled-call minus GPU tau exposes enqueue/backpressure
that cannot be called pure kernel service. Both relative-error checks must be
<=25% for the expert accounting gate.

For waits on actual slots, join9MiB DMA in submission order to checked copy-ticket
provenance. Bound potential GPU waiting by previous same-stream work and host
wait entry until DMA completion/next expert gather. This is an upper bound, not
an isolated stall measurement or a claim that every H2D wait is controllable.

Class-A service dominance screening compares frozen tau-derived BR/FCA return
wait delta with measured GPU-start-wait delta: at least50% without exceeding125%.
This necessary screening convention is not a fit to timings. Source-of-arrival
causality must remain unresolved when host scheduling, enqueue or staging terms
dominate and no separate intervention isolates them. A passing accounting gate
alone does not authorize an oracle. If B2 supports a revised model, validate
that model separately; otherwise publish the failed/limited diagnostic and stop.
