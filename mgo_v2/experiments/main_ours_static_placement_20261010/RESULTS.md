# `main_OURS` fixed-owner Static placement

R4 Qwen ShareGPT, input128, 32 decode forwards, prefetch OFF, native C++ Ready-First individual-expert execution. Static assigns `expert_id % 4` on GPUs 0/1/4/5. Each Static target starts after a cache reset and replays the exact original Near-captured route and teacher tokens; all four rank trace hashes and teacher tokens match the earlier cell. Static has **two unfiltered clean runs**; BR/CA/Near are the earlier **single** clean measurements, so their one-shot differences should not be treated as stable gains. The largest relative difference between the two Static repeats is 0.66%.

| Transport | C | B/rank | Seed | Earlier BR | Earlier CA | Earlier Near | Added Static mean [range] (s/token) |
|---|---:|---:|---|---:|---:|---:|---:|
| nvswitch | 60 | 8 | 20/7 | 0.4326 | 0.4378 | 0.4231 | 0.5282 [0.5277, 0.5287] |
| nvswitch | 60 | 8 | 14/5 | 0.4180 | 0.4324 | 0.4081 | 0.5037 [0.5034, 0.5040] |
| nvswitch | 60 | 16 | 22/3 | 0.5056 | 0.5112 | 0.5015 | 0.6219 [0.6212, 0.6226] |
| nvswitch | 60 | 16 | 16/3 | 0.4784 | 0.5010 | 0.4763 | 0.4680 [0.4678, 0.4682] |
| nvswitch | 60 | 64 | 13/0 | 0.5722 | 0.5872 | 0.5792 | 0.5523 [0.5513, 0.5534] |
| nvswitch | 60 | 64 | 20/5 | 0.5669 | 0.5810 | 0.5692 | 0.5478 [0.5468, 0.5488] |
| nvswitch | 30 | 8 | 20/7 | 0.4477 | 0.4543 | 0.4364 | 0.4497 [0.4487, 0.4508] |
| nvswitch | 30 | 8 | 14/5 | 0.4375 | 0.4433 | 0.4300 | 0.4351 [0.4346, 0.4355] |
| nvswitch | 30 | 16 | 22/3 | 0.5240 | 0.5298 | 0.5132 | 0.5083 [0.5081, 0.5085] |
| nvswitch | 30 | 64 | 20/5 | 0.5987 | 0.6109 | 0.5862 | 0.5749 [0.5744, 0.5754] |
| p2p_disabled | 30 | 8 | 20/7 | 0.4516 | 0.4581 | 0.4446 | 0.4489 [0.4476, 0.4502] |
| p2p_disabled | 30 | 8 | 14/5 | 0.4364 | 0.4430 | 0.4251 | 0.4340 [0.4332, 0.4348] |
| p2p_disabled | 30 | 16 | 22/3 | 0.5234 | 0.5298 | 0.5126 | 0.5085 [0.5083, 0.5086] |
| p2p_disabled | 30 | 64 | 20/5 | 0.5947 | 0.6048 | 0.5857 | 0.5762 [0.5753, 0.5770] |
| p2p_disabled | 60 | 8 | 20/7 | 0.4328 | 0.4432 | 0.4249 | 0.4205 [0.4205, 0.4206] |
| p2p_disabled | 60 | 8 | 14/5 | 0.4169 | 0.4295 | 0.4172 | 0.4077 [0.4064, 0.4091] |
| p2p_disabled | 60 | 16 | 22/3 | 0.5101 | 0.5241 | 0.5109 | 0.4854 [0.4853, 0.4854] |
| p2p_disabled | 60 | 64 | 20/5 | 0.5757 | 0.5839 | 0.5779 | 0.5567 [0.5560, 0.5575] |

All times are full TPOT including attention, in seconds per token. The original policy timings come from [SHAREGPT_RESULTS.json](../policy_gap_c60_20261008/SHAREGPT_RESULTS.json) and [FOLLOWUP_RESULTS.json](../policy_gap_c60_20261008/FOLLOWUP_RESULTS.json). Per-run Static values, provenance and raw receipt paths are in [RESULTS.json](RESULTS.json).
