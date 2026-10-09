# Does ready-first H2D/compute overlap help?

The C20 same-source, exact-output A/B passed. Both arms used identical
ShareGPT R4 B16/input512/output64 requests, Near placement, native C++ expert
execution, compiled layouts, full pinned expert source, prefetch OFF, and an
empty expert cache after warmup. The only intervention waited on all
current-layer demand copies after dispatch and before expert execution. No
global barrier or cache/policy change was added.

| C20 arm | TPOT repeat 1 | TPOT repeat 2 | Mean TPOT |
|---|---:|---:|---:|
| Normal ready-first | 502.210 ms | 503.575 ms | 502.893 ms/token |
| Wait for all H2D before expert | 588.347 ms | 588.143 ms | 588.245 ms/token |

Normal ready-first execution was **85.352 ms/token faster**; the serialized
arm was 16.97% slower. All 64 output sequences, per-rank H2D bytes, final
cache/ownership hashes, and per-rank expert-group counts matched exactly in
both repetitions. No measured repeat recompiled. The normal arm used
6,320–6,692 native waves per rank and made **zero** explicit wait-for-slot
calls; the serialized arm used 3,072 waves per rank and spent 5.29–6.08 s
per batch in host waits for demand copies. Fewer waves did not compensate for
losing overlap. The 3,072 waves are one per 48 routed layers × 64 generated
tokens; the normal arm splits work into more waves so it can execute groups
whose expert weights are already ready.

This directly rejects the suspicion that the separate copy stream and
ready-first schedule are merely a bookkeeping mistake in this condition.
They hide substantial H2D time despite the 9-MiB transfers. The A/B measures
the **net** effect of allowing expert/H2D overlap versus collapsing waves;
it does not isolate pure PCIe latency, because demand copies can still
overlap dispatch in the serialized arm. The normal arm's tiny explicit H2D
wait and this A/B together explain why halving H2D bytes need not halve TPOT.

The C50 serialized follow-up will test how much cache-capacity gain appears
when expert/H2D overlap is removed. The machine's earlier cross-job timing
drift remains a limitation; the two repeats within each A/B arm were stable.

Validated per-rank counters and exact-parity receipt are in
`OVERLAP_ABLATION.json`. Raw output tokens remain outside Git.
