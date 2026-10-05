# Owner amendment: implementation reliability and timing stability

The owner explicitly extends the active task to diagnosing and stabilizing both
profile failures and BR/LA primary timing variation before trusting a gain.
R4 on physical GPUs 0,1,4,5; local B128/B256; frozen decode64; BF16 remain fixed.
Two stable repetitions stop; at most one third, never unchanged-condition
reruns to select a favorable sample. Retain all observations.

Required evidence:
- Distinguish diagnostic-only CUDA instrumentation failures from uninstrumented
  E2E/TPOT variability; neither establishes the other's cause.
- Audit asynchronous H2D completion, pinned staging reuse, cache-slot overwrite,
  invalidation/promotion, and compute-use dependencies for races.
- Attribute BR and LA variation separately using matched workload counters and
  actual CPU/GPU intervals. Treat shared-resource and scheduling explanations
  as hypotheses until supported; CPU usage alone is not causal proof.
- Test any repair's relevant invariants, then run bounded matched BR/LA timing
  on the changed implementation. Do not pool different core fingerprints.
- Deliver supported cause(s), repairs, before/after E2E/TPOT and repeat ranges,
  remaining uncertainty, and a justified reliability/selection conclusion.
  No final positive gain claim from unstable or single-sample measurements.

Current evidence: R4_UNIQUE_VARIABILITY_ATTRIBUTION.json. Both policies vary;
B256 LA varies more. Work and copy counts are identical within each policy.
PROFILE_LOCKFIX_STALL.json shows the lock fix alone did not cure capture stalls.
The diagnostic-only untimed-event capture is a hypothesis test, not a repair
claim or a primary performance measurement.
