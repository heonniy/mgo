# AGENT TASK

1. Reuse the completed exact 2048-request pools; no GPU capture.
2. Reuse the frozen B128 MATH stress winner exactly.
3. Search B256 DP and BR placement seeds as specified in PLAN.md; all 2048
   requests are used, so do not invent a sample seed.
4. Implement persistent online replicas with actual cache-slot victims and
   expert-sized D2D creation accounting.
5. The replica decision may use current routing/cache state only; no suffix or
   future route oracle.
6. Evaluate the predeclared 24-policy grid for BR and CA at B128 and B256.
7. Compute the B256 free one-replica oracle only as an upper-bound reference.
8. Report critical rows, H2D, D2D, peer bytes, hit rate, replica creation/reuse
   and victim statistics.
9. Mark the best <=2% H2D-growth policy separately and provide the Pareto
   frontier. Do not combine H2D and compute into an arbitrary lambda score.
10. Commit results and stop. No physical GPU timing automatically follows.
