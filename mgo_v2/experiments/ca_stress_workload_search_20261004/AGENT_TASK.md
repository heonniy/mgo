# AGENT TASK — CA stress workload search

Start only after the active timing-stability diagnostic finishes, regardless of
PASS/FAIL.

1. Read README.md and PLAN.md.
2. Keep substitution OFF, Gate W128, decode256, exact experts.
3. Confirm from the live policy code that BR and CA share identical per-event
   balanced mandatory-miss quota slots; retain the max-min <=1 assertion.
4. Reuse the existing 512 exact traces.
5. With all 8 H100s, extend MATH and ShareGPT candidate pools to 2048 requests
   each when the existing dataset/filter permits; do not recapture the first 512.
6. Run the Stage-A sample_seed x dp_seed routing prescreen on CPU.
7. Run exact Gate cache replay for the retained candidates, four cache ratios,
   and BR seeds [7,19,42,73,99,131,181,251].
8. For every dataset x R{4,8} x B{8,16,32,64} x cache{30,40,50,60}, select and
   report the structural best sample/dp pair and the best-observed BR seed.
9. Validate request manifests, rank balance, per-event miss quotas, route hashes,
   cache capacity and final state.
10. Commit results and stop. Do not automatically launch physical policy timing.

The timing diagnostic result does not cancel this resource-search packet.
