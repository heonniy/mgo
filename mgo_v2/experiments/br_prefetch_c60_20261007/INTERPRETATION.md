# BR prefetch / rank critical-path interpretation

This document is provisional until STATUS.json is PASS for all three batches.
Numerical tables are in RESULTS.md and RANK_DIAGNOSTICS.md; full summaries
include every rank rather than only the slowest or fastest rank.

## What this comparison can establish

The comparison keeps MAIN capacity, physical slots, streaming demand-H2D,
ready-first execution and T2 synchronization identical. OFF suppresses only
speculative prefetch submissions. Each arm starts cold before prefill and
runs 256 decode forwards. B is local batch; global batches are 32/64/256.

There is one clean primary per condition and a separate instrumented run.
Each instrumented run must reproduce its primary tokens, final cache/role
state and payload/H2D byte counts. OFF and ON are native greedy generations:
BF16 rank-partial accumulation can change generated tokens and subsequent
routes. Thus a timing difference is an observed runtime difference, not an
isolated causal prefetch effect on an identical frozen workload.

## Runtime sequence and the meaning of communication time

Each decode MoE layer gathers routes and prepares placement/layout, submits
required H2D on its copy stream, then submits dispatch all-to-all. T2 waits
for local dispatch/unpack completion before optional next-layer prefetch.
Ready-first expert execution gathers rows and submits a kernel per expert,
weights outputs, and records slot-use events. Rank-local weighted partials
are returned with all-to-all and combined at the origin rank.

There is no added global barrier before dispatch or after expert execution
in either decode arm. Routing collectives, dispatch dependencies, and return
collectives still couple ranks. In particular an early rank's return span
can include waiting for a late rank to finish expert execution. A large
return span therefore does not by itself demonstrate bandwidth saturation.

Read three entry skews together: dispatch, expert, and return. If expert
entry is nearly aligned but return entry is dispersed, much of the
divergence developed in expert execution / partial preparation. Compare
expert groups and token-expert rows with those spans. Expert time is a
current-stream completion span including host submission gaps, not pure
GEMM. Its own-thread CPU time provides separate evidence of executor cost.

## B8 evidence

Primary TPOT is 0.643995s OFF and 0.707467s ON (ON 9.86% slower).
Peer payload is almost unchanged (8.181 versus 8.175 GiB per 256 steps),
whereas H2D increases from 2025.562 to 2259.281 GiB. ON issues 96,256
prefetches, uses 72,272 and wastes 23,984. Mandatory fetches decline by
69,664, less than useful prefetches, leaving 26,592 additional total copies.
These controller counters include prefill; speculative copies occur in
decode. The low 26.40% OFF/ON token agreement prevents treating that count
difference as an identical-route counterfactual.

For OFF, mean expert-entry skew is 0.266ms/layer and return-entry skew is
2.550ms/layer; median expert-span spread is 2.129ms/layer. Ranks 0/1 spend
about 345ms/step in expert execution and 56–57ms in return, whereas rank 3
spends 318ms in expert execution and 89ms in return. Own-thread expert CPU
time is close to these expert spans. ON also adds roughly 41–42ms/step of
prefetch-controller work per rank in its diagnostic.

This supports per-expert host/executor work and rank finishing-time skew as
substantial contributors in B8. It does not support declaring pure network
bandwidth the dominant bottleneck, nor does it establish that H2D is free.
Most primary ready-first events already find experts ready; copies can be
hidden behind host work and dispatch. Instrumentation can hide more waits.

Decode-only copy service was recovered without another GPU run: the saved
prefill byte boundary identifies the initial copy prefix, because scheduler
trace insertion and copy-byte accounting share a lock and prefill completes
required transfers before decode. Remaining copy bytes match primary decode
bytes exactly. B8 rank 1/GPU1 has the slowest copy tail: p99 0.371ms OFF and
1.053ms ON, versus ON p99 0.413–0.417ms on other ranks. This is a real
rank-dependent service-tail observation, compatible with contention, but
does not identify its cause. Primary explicit ready-first wait counts are
zero on all ranks, so it is not evidence that this tail dominates TPOT.

## B16 evidence

Primary TPOT is 0.753269s OFF and 0.813985s ON (ON 8.06% slower).
Peer payload rises only 0.13%; H2D rises 1.67%. Prefetch usefulness is
93.18%, but the primary has only one explicit ready-first wait OFF and zero
ON across all ranks. High usefulness therefore does not imply a comparable
amount of exposed time saved. OFF/ON token agreement is 27.09%.

OFF rank 0 has 444ms/step expert span and 46ms return, versus rank 2's
402ms expert and 94ms return. Expert-entry skew is 0.340ms/layer and return
entry skew is 2.957ms/layer. ON skews are 0.347ms and 2.669ms, respectively.
Own-thread expert CPU time is 401–430ms/step across these arms/ranks, again
showing substantial host/executor work. The slowest expert rank belongs to
the maximum-expert-count set in 83.0% OFF / 93.7% ON events, versus only
53.5% / 58.4% for maximum token rows. This supports counting per-expert
invocations in a load model, rather than relying only on token count.

ON copy-service p99 is 1.02–1.12ms across all ranks; OFF is 0.23–0.39ms.
The service tail is worse under ON, but its lack of exposed ready-first
waits means the current data does not identify H2D as the dominant blocker.
No single physical GPU is uniformly the slow H2D rank across B8 and B16.

## Placement implication to test, not a measured winner

Minimizing communication bytes alone is not the same as minimizing TPOT.
The useful target is the maximum rank completion time at return: include
per-expert invocation cost, token-row work and exposed fetch dependencies,
then favor locality among similarly loaded placements. Count distinct
expert invocations as well as rows, particularly at small batches where
each expert receives few tokens. Cache-resident placement restricts which
work can be moved through admission alone; a balanced future-miss assignment
cannot necessarily rebalance all currently resident expert work.

This is a hypothesis for a load-aware/locality-tiebreak strategy, not proof
that current LA, CA, FCA or Near is fastest. Only BR was requested in the
reduced scope. No additional placement or synchronization experiment is
launched by this analysis.

## Non-additive and unmeasured quantities

Diagnostic MoE excludes attention outside MLP but includes routing,
placement, dispatch, fetch dependencies, expert execution, return/combine
and host gaps. It is a separate instrumented sample and can exceed clean
TPOT; never subtract it from primary TPOT to estimate attention.

Unprefixed copy service in SUMMARY.json includes prefill; decode-prefixed
fields exclude the validated prefix. Both overlap execution. A tiny exposed
wait does not bound all H2D effects. Entry skews
use same-host CPU timestamps, not NCCL GPU kernel timestamps. No network
bandwidth counter, isolated GEMM kernel profile, causal policy speedup or
repeat-stability guarantee is claimed.
