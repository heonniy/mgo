# OURS large-cell bounded confirmation

All three repeats pass frozen manifest/raw-clock, finite-logit, Near policy, cold expert cache,1843 physical slot and no-compilation checks. Timing remains UNSTABLE. Retain all samples; no stable headline selection or additional automatic repeats.

- TTFT: median 26.584921701s; range [26.405345780, 28.784069675]s; spread 8.727%.
- TPOT: median 1.256196380s; range [1.119625152, 1.288150259]s; spread 13.799%.
- E2E: median 107.558812108s; range [97.121306298, 107.924441605]s; spread 10.368%.

Selected runtime remains H0/full-pinned, V3P2/T2, LA_CA_NEAR;216GiB total rank-private pinned expert backing. Warmup uses disjoint prompts and every measured repeat starts with cleared expert state; measured prefill residency continues into decode. No output timing sample is removed as noise.
