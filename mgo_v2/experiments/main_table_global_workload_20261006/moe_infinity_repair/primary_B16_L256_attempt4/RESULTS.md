# Repaired MoE small cell

All3 repeats pass manifest/output/raw-clock, actual KV release, cold expert cache, EAM, priority eviction and per-GPU budget checks. Overall timing UNSTABLE because TTFT exceeds5% spread. All samples retained.

- TTFT: median 4.916001564s; range [4.912147844, 5.250308408]s; spread 6.728%.
- TPOT: median 3.050296530s; range [3.025173636, 3.086803592]s; spread 2.018%.
- E2E: median 197.080829236s; range [195.501940647, 199.718934678]s; spread 2.136%.

One bounded confirmation follows the existing serial queues. No outlier removal or automatic repeat loop.
