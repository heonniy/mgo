# AGENT TASK — fetch-matched B32 + 2-A2A

Do not interrupt the currently running ca_stress_physical_validation_20261004
R4 work. Start this packet only after all of those scientific processes exit
and their results are committed.

1. Read README.md and PLAN.md.
2. Reuse the completed 2,048-request ShareGPT exact-route pool.
3. Search R4/B32/cache30 and R8/B32/cache30 for sample_seed, dp_seed and BR seed.
4. Reject any candidate where BR and CA total fetches or H2D bytes differ.
5. Instrument per-rank send/recv bytes and select one R4 and one R8 winner using
   the critical-rank-first score in PLAN.md.
6. Build a common frozen-route, teacher-forced physical replay so BR/CA/A3/A2
   consume identical tokens, expert IDs and routing weights.
7. Implement A2 by removing routing-weight A2A, returning unweighted expert
   outputs, and applying frozen weights on the source before combine.
8. Pass the untimed A2 numerical/call-count correctness gate before timing.
9. Physical timing is Env2 only:
   - R8 on GPUs 0..7;
   - R4 on GPUs 0,1,4,6.
10. Run BR/A3, CA/A3, BR/A2, CA/A2 with two repeats first; add one third repeat
    only under the frozen 2%-5% rule.
11. Keep controller/Hungarian, counters, compilation and heavy monitors outside
    MEASURE.
12. Commit compact results and stop. No automatic extension.
