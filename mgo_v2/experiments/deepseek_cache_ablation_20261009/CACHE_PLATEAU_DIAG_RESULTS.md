# Why C20 to C50 saves H2D bytes but little TPOT

The answer from the separate C20/C50 diagnostic is that **most H2D work is
hidden behind other decode work**. Prefetch is OFF, but demand H2D is still
submitted on a separate CUDA stream, while the native executor processes
experts whose cache slots are already ready. Fewer transfers therefore reduce
PCIe traffic without removing the same amount of exposed decode latency.

Both diagnostic runs used the frozen ShareGPT B16/input512/output64 target,
one warmup and one target, on GPUs 0/1/4/5. All four ranks produced exactly
the same full output tokens and total H2D bytes as their respective C20/C50
three-repeat primary runs. These diagnostic timings do not replace the
primary table.

| Metric | C20 | C50 |
|---|---:|---:|
| Primary TPOT median, ms/token | 287.904 | 281.710 |
| Instrumented TPOT, ms/token | 297.233 | 292.413 |
| Decode H2D, GiB summed over four ranks | 1,397.5 | 878.4 |
| Largest rank's expert-not-ready calls over 63 decode intervals | 306 | 5 |
| Largest rank's **exposed CUDA-stream H2D wait**, ms/token | 0.327 | 0.037 |
| Largest rank's host wait for H2D submission, ms/token | 0.045 | 0.028 |
| Route-to-dispatch host span, rank range in ms/token | 58.8–72.8 | 58.5–86.4 |
| Dispatch span, rank range in ms/token | 28.0–30.6 | 23.5–27.6 |
| Expert span, rank range in ms/token | 85.3–93.7 | 72.6–90.2 |
| Return and combine span, rank range in ms/token | 76.8–82.2 | 79.1–96.6 |

Increasing capacity cut actual **decode** H2D bytes by 37.1%, not merely
prefill traffic. The largest directly exposed H2D wait at C20 was only
0.327 ms/token, approximately 0.11% of the instrumented TPOT; at C50 it was
0.037 ms/token. Expert work, route/layout preparation, dispatch, and
return/combine still occupied tens of milliseconds each per rank and token.
Peer dispatch/return volume in the primary runs stayed near 9 GiB each per
batch at both capacities. These observations explain why saving hundreds of
GiB of DMA does not yield a proportionate TPOT improvement. The primary
TPOT improvement was 6.19 ms/token, or 2.15%, from C20 to C50.

The H2D wait figure measures the explicit dependency on the main CUDA stream,
not total DMA duration or possible PCIe/HBM contention. The dispatch and
return spans include packet work and waiting, so they are not pure network
time; the host span is not pure controller CPU time. The spans can overlap
and must not be added into a synthetic TPOT decomposition. Instrumentation
raised TPOT by 3.2% at C20 and 3.8% at C50. Moreover, C20 and C50 generated
different later tokens for some requests, so this pair cannot assign the
remaining few milliseconds to a single unchanged component. The conclusion
supported by this diagnostic is narrower and strong: **direct H2D waiting is
not the dominant TPOT bottleneck at either tested capacity**.

[CACHE_PLATEAU_DIAG.json](CACHE_PLATEAU_DIAG.json) contains the per-rank
aggregate spans and validation flags. The diagnostic-only raw per-step records
and token IDs remain outside Git in `dca_plateau_c20_b16_r1_v1` and
`dca_plateau_c50_b16_r1_v1` under the guarded job output directory.
