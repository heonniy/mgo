# Fetch-matched B32 CA + 2-A2A physical validation

Queued after the currently running CA-stress physical packet fully exits.

This packet asks two narrow questions:

1. Can we choose real ShareGPT request subsets / DP partitions / BR seeds where
   CA reduces inter-GPU communication **without changing total expert fetches or
   H2D bytes** relative to BR?
2. At local batch 32, does removing the routing-weight All-to-All (3 A2A ->
   2 A2A per MoE layer) make CA's communication reduction translate more
   directly into physical decode latency?

Exactly one best R8 cell and one best R4 cell are selected. No substitution,
replication, LRU, quality run, or broad physical sweep.
