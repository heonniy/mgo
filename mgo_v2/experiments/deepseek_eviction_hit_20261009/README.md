# DeepSeek eviction hit-rate replay

The prior DeepSeek D1 fixed-route study left four validated rank-local
`route_rank*.npz` files in the results directory. This follow-up consumes
those same files **on CPU only**. No model execution or GPU routing capture
was repeated. Each file's SHA-256 is checked against the passing D1 physical
replay receipt before calculation.

The frozen source is ShareGPT, R4, B16 per rank, input512 and 63 decode
forwards. Each event represents a distinct `(layer, expert)` request. The
gate score uses the last 128 global router-probability rows for the current
layer, matching the recorded prefill suffix and rolling decode window. All
five policies use the same BR admission shuffle and seed42: gate-score,
LFU-reset, LFU-cumulative, LRU-reset and LRU-cumulative. LFU adds one count
per active expert per layer/step. `cumulative` retains frequency/recency
metadata after an expert is evicted; `reset` discards it. Current active
experts cannot be evicted. DeepSeek's physical cache budget reserves two
non-MAIN slots per rank; only MAIN slots are replayed.

| Model | Capacity | Gate-score | LFU (history retained) | LRU (history retained) |
|---|---:|---:|---:|---:|
| DeepSeek-V2-Lite | C30 | 26.07% | 0.00% | 0.00% |
| DeepSeek-V2-Lite | C60 | 57.47% | 1.69% | 0.00% |

These are pre-admission **distinct-expert hit rates** in decode. Prefill
fills the cache and is excluded from the numerator and denominator. They are
CPU counterfactuals on a route trace captured under C20; C60 was not
physically timed in this packet. The BR owner map may differ from the live
Near controller and from the earlier Qwen replay. Consequently, the
two-model plot is a side-by-side within-model policy comparison, not a claim
that the absolute Qwen and DeepSeek hit rates are controlled for identical
workloads. Qwen uses input256/256 decode steps, while DeepSeek uses
input512/63 decode steps. All input and output bytes remain outside Git.

`RESULTS.json` records both reset and history-retained variants and source
hashes. The replay checks hit+miss conservation, occupied slots versus net
admissions, bounded reloads, and exact LRU-reset/LRU-cumulative equivalence.
