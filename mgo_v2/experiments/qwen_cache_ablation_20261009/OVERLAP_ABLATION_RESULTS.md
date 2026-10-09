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

The C50 serialized follow-up and a same-source normal C50 control also passed
with exact request, all-token, H2D-byte, cache-hash and expert-group parity.
Their two-repeat mean TPOTs were **504.952** and **484.263 ms/token**,
respectively: disabling expert/H2D overlap cost **20.688 ms/token** at C50.
The normal C50 arm used 5,982–6,072 native waves/rank, versus 3,072 in the
serialized arm. The serialized host wait was 1.65–2.17 s/batch/rank.

| Cache | Normal ready-first, mean ms/token | Wait-all-H2D, mean ms/token | Serial penalty |
|---:|---:|---:|---:|
| C20 | 502.893 | 588.245 | 85.352 ms/token |
| C50 | 484.263 | 504.952 | 20.688 ms/token |

Within each cache capacity, changing only the wait behavior creates a
controlled same-output comparison. Across capacities, later generated routes
can differ, so the larger C20→C50 gain in the serialized arms is not a pure
fixed-route estimate of copy-removal headroom. It does show why the normal
runtime can halve copy bytes but improve TPOT modestly: ready-first execution
already hides much of the transfer service. Even at C50, normal execution
uses about two waves per routed layer, so fewer misses do not collapse the
remaining scheduling and expert work. The machine's earlier cross-job timing
drift remains a limitation; the two repeats within each arm were stable.

Validated per-rank counters and exact-parity receipts are in
`OVERLAP_ABLATION.json` and `OVERLAP_CAPACITY.json`. Raw output tokens remain
outside Git.
