# AGENT TASK

1. Reuse the completed 2048-request MATH and ShareGPT exact pools.
2. Do not capture new routes.
3. Freeze R=8, local B=128, cache=60%, Gate W128, substitution OFF, decode256.
4. Run the bounded sample/placement/DP stress search from PLAN.md.
5. Selection must use BR one-copy structural metrics only.
6. Exact-replay the retained triples and deterministic CA.
7. Compute both temporary one-replica oracle variants.
8. Freeze the final winning manifest and top alternatives.
9. Report BR vs CA communication headroom and BR/CA replication compute
   headroom separately.
10. If the 10% oracle gate fails, stop. If it passes, commit results and stop
    for owner review; do not automatically implement or time a real replica
    policy.
