# Fetch-matched B32 CA + 2-A2A fast physical pivot screen

Queued after the currently running CA-stress physical packet fully exits.

This packet asks two narrow questions at **local B32 / decode64**:

1. Under exactly matched BR/CA expert-fetch and H2D work, how much can CA reduce
   (a) aggregate peer bytes and (b) per-event critical-rank All-to-All load?
2. Does removing the routing-weight All-to-All (3 A2A -> 2 A2A per MoE layer)
   make either placement objective translate more directly into physical decode
   latency?

Reuse the first 64 decode steps of the existing 256-step exact ShareGPT traces;
do not recapture the model. Select Peer-best and Critical-best independently
for R4 and R8, deduplicating identical winners.

This is an exploratory pivot screen: frozen routing, Env2 only, single-shot
timing first, one confirmation pair only for >=1% positive CA gain. No
substitution, replication, LRU, quality run, or broad physical sweep.
