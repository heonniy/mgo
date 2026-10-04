# AGENT TASK — fast fetch-matched B32 + 2-A2A pivot screen

Do not interrupt the currently running ca_stress_physical_validation_20261004
R4 work. Start this packet only after all of those scientific processes exit
and their results are committed.

1. Read README.md and PLAN.md.
2. Reuse the completed 2,048-request ShareGPT exact-route pool, taking only the first 64 decode steps of each existing 256-step trace; do not recapture.
3. For R4/B32/cache30 and R8/B32/cache30, prescreen 512 sample seeds x 64 DP
   seeds and retain top-64 for each of two independent objectives:
   aggregate peer-byte reduction and per-event critical-rank reduction.
4. Exact Gate-replay the deduplicated candidates across BR seeds
   [7,19,42,73,99,131,181,251]. Reject any BR/CA pair where total fetches or
   H2D bytes differ.
5. Select **two winners per R**:
   - Peer-best: maximize aggregate peer-byte reduction.
   - Critical-best: maximize sum over A2A events of the per-event max-rank
     send+recv reduction.
   Deduplicate if the exact same sample/DP/BR seed wins both.
6. Build a common frozen-route, teacher-forced physical replay so BR/CA/A3/A2
   consume identical tokens, expert IDs and routing weights.
7. Implement A2 by removing routing-weight A2A, returning unweighted expert
   outputs, and applying frozen weights on the source before combine.
8. Pass the untimed A2 numerical/call-count correctness gate before timing.
9. Physical timing is Env2 only. Run **R4 first**, then R8:
   - R4 GPUs 0,1,4,6.
   - R8 GPUs 0..7.
10. Fast screen: for every unique Peer-best/Critical-best workload, run
    BR/A3, CA/A3, BR/A2, CA/A2 **once each**. No full 64-step schedule warmup per
    MEASURE; use only the one-time compile validation plus short 8-16-step
    readiness warmup per (R,A3/A2).
11. Selective confirmation: if a fixed BR-vs-CA pair shows >=1% positive CA
    gain in any primary timing metric, run **one additional confirmation pair
    only** (BR once + CA once). Never add a second confirmation repeat.
12. Keep controller/Hungarian, counters, compilation and heavy monitors outside
    MEASURE. Any recompile inside MEASURE invalidates that run.
13. Commit compact raw results/counters and stop. No automatic extension.
