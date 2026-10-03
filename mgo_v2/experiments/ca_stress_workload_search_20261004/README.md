# CA-favorable workload / DP-partition search

Start **only after** `timing_stability_numa_20261004` finishes its currently
authorized diagnostic sequence. This follow-up runs regardless of whether the
timing harness passes or fails, because the main search is frozen-route
resource accounting rather than E2E timing.

Goal: find reproducible real-request subsets and balanced DP rank assignments
where CA yields the largest communication reduction relative to BR.

This is explicitly a **communication-stress / best-case characterization**, not
a representative-random-workload result.

Frozen method choices:
- Gate eviction only;
- substitution **OFF**;
- exact experts only;
- decode256;
- BR and CA only;
- no CA-rep in this packet.
