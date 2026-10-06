# DeepSpeed large-cell bounded confirmation

All three repeats pass frozen workload/raw-clock, cold-parameter, GPU KV, finite-logit and byte-cap checks. Timing remains UNSTABLE. Every sample is retained; no slow-sample filtering, stable headline selection or automatic additional repeats.

- TTFT: median 7.927774667s; range [5.692961276, 8.420578384]s; spread 37.125%.
- TPOT: median 4.825910471s; range [4.696273746, 6.305955925]s; spread 30.509%.
- E2E: median 311.960134312s; range [304.285824369, 402.968184522]s; spread 29.047%.

TPOT samples4.696274/6.305956/4.825910s show the second repeat is slower. In repeat2,50of63 intervals exceed5.5s on every rank, versus zero in repeat1. This is sustained within-run variability, not one isolated delay. Collective synchronization means matching rank intervals cannot identify a particular causal rank. Resource/clock receipts do not prove the underlying cause.
