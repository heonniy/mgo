> **Repetition override:** `REPETITION_AMENDMENT.md` supersedes all earlier three-repeat / two-extra-repeat instructions below. Use two clean samples, at most one conditional third, independently per environment/policy.

# R4 extension agent task

Run only after the existing R8-best physical validation has fully exited and
its results are committed.

1. Use ShareGPT_R4_B8_c30.json from the completed CA stress search.
2. Fix physical GPUs to 0,1,2,3.
3. Freeze sample_seed=205, dp_seed=4, BR seed=73.
4. Keep local B8, cache30, Gate W128, substitution OFF, decode256.
5. Use the validated BOUNDARY/fixed-affinity harness.
6. Keep Hungarian/controller computation outside MEASURE.
7. Run BR vs CA in Env1 and Env2, 3 repeats each, counterbalanced as specified.
8. Verify COUNTERS reproduce the frozen ~30.1124% chosen-seed peer reduction
   direction and report the eight-seed 29.6355%--30.1124% resource range.
9. Commit compact results and stop. Do not add more R/B/cache cells.
