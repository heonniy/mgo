# Repaired MoE large cell

PASS: all three primaries satisfy request/token/raw-clock, cold residency, EAM, priority eviction, exact byte caps, KV release and <=5% timing spread checks. No additional repeats.

- TTFT: median 9.650224808s; range [9.523654747, 9.674713595]s; spread 1.571%.
- TPOT: median 3.696201790s; range [3.637762132, 3.761963579]s; spread 3.358%.
- E2E: median 242.535426368s; range [238.829239148, 246.527360193]s; spread 3.173%.

Every repeat uses the same warmup EAM snapshot and starts with zero expert residency. All3 have3072 EAM calls,4737472 candidate submissions, positive priority evictions, and global peak charged expert bytes17392730112. Warmup and all primaries have identical PyTorch peak allocated bytes; KV weakrefs are released after each batch. Total NVML HBM includes non-expert allocations and is separately reported in audit.json.
