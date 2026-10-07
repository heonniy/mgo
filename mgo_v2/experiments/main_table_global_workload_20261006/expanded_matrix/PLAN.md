# Owner expanded R4 main table, 2026-10-07

Chat supersedes historical two-cell/repetition restrictions. Run four systems:
repaired MoE-Infinity, stock DeepSpeed ZeRO-3 CPU parameter offload,
llama.cpp synchronous native batch, and repaired OURS LA_CA_NEAR H0/full-pinned
with prefill-optimized and prefill-layout-fast. No decode/runtime retuning.

12 cells: local B16/B32/B64 x input256/512 x cache30/60%; global requests=4B.
Output64 greedy tokens, continue after EOS, BF16. GPUs0,1,4,5 only.
Same frozen ShareGPT-long IDs across systems; disjoint warmup conversations.
Dynamic expert/KV/history state reset per primary; static weights retained.
No separate profiling. One GPU job at a time, host available>=384GiB before
launch and abort below96GiB. Never touch GPUs2,3,6,7 or foreign processes.

Exactly5 primary repeats following warmup per cell/system, 240 primaries.
The preceding four-repeat OURS packet remains separate and is not reused.
Before observing new results, freeze selection: enumerate all 10 triplets;
for each compute (max-min)/mean for TTFT, TPOT and E2E. Minimize maximum
of these three spreads, then sum of spreads, then lexicographic repeat IDs.
Use ONE selected triplet for all metrics; not fastest values or separate
per-metric selections. Report selected mean/sample SD and IDs, all5 raw values,
all5 mean/sample SD/range and selected spreads. If selected worst spread>5%,
label UNSTABLE; no extra repeats or claim that excluded repeats are invalid.
Selection is descriptive, not an unbiased estimate of unconditional latency.

C30=1843 slots; C60=3686 slots; one slot=9MiB, budgets global.
OURS per-rank slots include2 prefetch slots. Infinity uses exact byte limits.
DeepSpeed doubles native live/prefetch knobs at C60 and checks actual all-
parameter residency against its per-rank cap; it is not expert-selective LRU.
llama uses floor(slots/128) GPU expert layers (14/28), other expert layers CPU;
all attention and KV stay GPU. Report static unused budget due layer granularity.
Its native runner completes all prefill before any decode, synchronizes each
whole-batch token step, and retains each sequence's greedy token outputs.

TTFT=global release to all first tokens; E2E=release to all64 tokens;
TPOT=(E2E-TTFT)/63. No asynchronous tail TPOT in this packet.
Archive provenance and receipts; incremental commits. Fix implementation or
budget failures before affected primaries; preserve failed attempts and continue
independent valid jobs. Do not silently shrink batch, precision or budget.
