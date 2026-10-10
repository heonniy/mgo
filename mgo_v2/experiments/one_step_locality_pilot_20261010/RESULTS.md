# One-step model pilot: local execution, peer bytes and critical-rank load

**Finding.** No tested placement improved the measured first decode step over
BR on this frozen route. More local expert rows did not necessarily reduce
token/rank packets. The policy that reduced peer bytes most also increased
the busiest rank's expert-row load. LA_CA reduced both peer bytes and that
load, but its four timings were noisy and did not establish a gain.

## Scope and method

Qwen3-30B-A3B-Instruct-2507, ShareGPT, physical GPUs 0/1/4/5, rank-local
B16, input512, C30, prefetch OFF, full-pinned CPU expert source, native
prefill, compiled dense routing and strict hit-then-miss grouped `new_OURS`.
Every measured generation contains one prefill forward and **one** decode
forward (48 MoE layers). Prefill placement is BR in every arm. The next-token
input and all 48 decode-layer selected IDs, routing weights and gate
probabilities are frozen from a BR model capture and replayed under BR, Near,
CA_NATIVE and LA_CA. Expert weights, H2D copies, NCCL, grouped GEMM, dense
model computation and KV attention remain physical. The frozen route makes
this a controlled model-path pilot, not a free-generation or dataset-average
result.

Four clean target generations per arm followed one disjoint warmup; a
separate instrumented generation collected resource counts. All target
measurements are retained. Every diagnostic reported the identical decode
route SHA256 `eae84e151531ce21f69b3e471bbb7a64afdfc9eca6ac8dc226d9d0832dea5ce7`.
All arms had the same rank-local first token from prefill, and each
instrumented run matched its own last clean run's tokens and cache state.
The earlier live BR/Near attempt is excluded from the fixed-route comparison:
BF16 execution order changed later layer routing even within the first
decode forward.

## Physical results

`Step` is the single clean decode-forward latency: median [full range] over
four unfiltered repeats. The local percentage counts expert-token rows
executed on their origin rank, not rank-packet locality. Peer bytes are the
sum of off-rank dispatch and return payloads over all four ranks. Max H2D
copies and max expert rows are the largest per-rank totals over the 48
decode layers. They are resource counts, not isolated service times.

| Decode placement | Step ms | Local expert rows | Off-rank packets | Peer MiB | Max-rank H2D copies | Max-rank expert rows |
|---|---:|---:|---:|---:|---:|---:|
| BR | 246.239 [243.816, 266.188] | 24.10% | 8,407 | 65.872 | 422 | 6,487 |
| Near | 246.209 [245.532, 246.737] | 29.51% | 8,430 | 66.052 | 422 | 6,438 |
| CA_NATIVE | 248.556 [248.063, 248.774] | 34.02% | 7,777 | 60.936 | 422 | 6,954 |
| LA_CA | 263.882 [246.871, 275.794] | 25.85% | 8,254 | 64.673 | 422 | 6,384 |

All four policies used exactly 1,610 decode demand copies distributed as
`[422, 409, 394, 385]` over GPUs 0/1/4/5; each copy is one 9 MiB expert.
Their identical miss counts rule out a reduced-fetch explanation for this
step. The separate diagnostic's maximum rank exposed H2D waits were
7.53/7.64/7.39/7.41 ms for BR/Near/CA_NATIVE/LA_CA; a single instrumented
pass does not prove equal physical H2D latency. Grouped expert execution
and preparation scopes were also close, with maxima of
39.34/39.03/39.40/38.97 ms. Diagnostic spans include host gaps and peer
waiting and cannot be added to the clean step times.

Near raised the local expert-row share by 5.41 percentage points and slightly
reduced maximum rank expert rows, yet created 23 additional remote
token/rank packets. Coalesced transport sends a token once per destination
rank, even if it needs multiple experts there; an extra local expert row
does not guarantee an entire remote packet disappears. Thus peer payload
increased 0.27% and no step-time gain appeared.

CA_NATIVE removed 630 remote packets and 7.49% of peer bytes. Its
maximum rank expert rows rose 7.20%, and the instrumented placement-controller
CPU span rose from about 6.9 to 9.0 ms for the step. Its measured step
median was 0.94% longer. The clean BR range includes a slower outlier, so
this small median difference is descriptive rather than a stable regression.

LA_CA is the closest resource-feasible candidate: 1.82% fewer peer bytes,
1.59% fewer maximum-rank expert rows and unchanged H2D copy counts. Its
packet-aware controller took about 8.9 ms versus BR's 6.9 ms in the separate
diagnostic, and the clean step samples were 275.794, 269.317, 246.871 and
258.446 ms. This short pilot cannot separate controller overhead from
scheduling noise well enough to assign the full timing gap, but it provides
**no measured step-time benefit** for this implementation.

## Interpretation boundary

This one-step test answers whether the currently implemented placements
produce an immediate physical-model gain on one controlled route. They do
not. It does not rule out gains on a different route, batch size or
communication fabric, and four one-step repetitions cannot establish a
general TPOT ranking. The stronger target for a follow-up is to reduce
**remote token/rank packets**, keep maximum rank H2D and expert work bounded,
and avoid spending the byte saving in placement-controller work. An explicit
all-rank barrier would not change the packet counts above.

Recompute the table with `summarize.py`. Small provenance and metrics are in
`SUMMARY.json`; complete rank receipts, resource logs and frozen route tensors
remain under `/home/hwlee/mgo-results/single_step_locality_pilot_20261010/`.
Each guarded job stopped only the four owned GPU 0/1/4/5 model loads and
restored them after confirming no measured GPU process remained. GPUs
2/3/6/7 were untouched.
