# AGENT TASK — physical CA stress validation

Execute exactly the one frozen cell in PLAN.md.

1. Stop the project's resident GPU load workers and verify all eight H100s are
   available.
2. Reuse the selected manifest
   `ShareGPT_R8_B8_c30.json` from commit 0b6da6e.
3. Freeze BR seed 42 and CA deterministic assignment.
4. Keep Gate W128, cache30, substitution OFF, decode256.
5. Use the validated BOUNDARY/fixed-affinity timing harness.
6. Run PLAN -> COMPILE/WARMUP -> MEASURE -> COUNTERS separately.
7. MEASURE must contain no Hungarian/controller work, no PSS/smaps monitor,
   no profiler, no detailed counters, and no compilation.
8. Run BR/CA in both Env1 and Env2, three clean repeats each, in the prescribed
   counterbalanced order.
9. Validate hashes and compare COUNTERS against the frozen CPU resource
   expectation, especially the ~21% peer reduction.
10. Commit compact results and stop. Restore resident GPU workers only after all
    scientific processes exit.
