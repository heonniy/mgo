# Owner amendment: decode64, TPOT/E2E report

The owner removed c30/s1 and shortened the sole c60/s0 arm to decode64.
Both previous drivers and their in-progress PLAN processes were stopped before
any timing sample. The new queue contains only c60_s0_h64.

Keep MATH/R4/local-B64/cache60/substitution-OFF, GPUs0,1,4,5, BR/CA,
pinned H2D, and current/coslot/coslot-active. Each condition still uses
COMPILE x1, MEASURE x3, COUNTERS x1; Env1 then conditional stable Env2.
Generate/validate/freeze a 64-step reference PLAN per policy outside timing.
Never confuse 256 and 64 plans: launcher discovery checks decode_steps and
new output/compile identities include h64. For 65x48=3120 layer events,
current has 9360 A2A calls/rank; coslot has 6240; active-peer uses 6240 rounds.
These counters remain internal validation, not owner-facing performance metrics.

Report only TPOT and E2E medians and full ranges, and BR/CA differences for
those metrics. No communication/H2D counter tables in the owner readout.
Raw correctness and resource evidence is retained for reproducibility.
