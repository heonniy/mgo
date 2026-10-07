# C60 CPU counterfactual

Owner requests C60 follow-up. Reuse exactly the existing C30 source traces
for B8/B16/B64, input256, prefill then256 decode steps, empty initial cache.
Five policies, BR/seed42, prefetch OFF, LFU distinct-event frequency unchanged.
All3686 slots are MAIN:922/922/921/921. No GPU model run or timing claim.
C30 Gate physical parity remains prior evidence; C60 only asserts replay
conservation, fixed trace hashes and LRU reset/cumulative equivalence.
Save all per-step counts and compare against the existing C30 summaries.
