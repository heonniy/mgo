# First native headline attempts

Attempt1 failed on an uncompiled expert-kernel input/arena-view specialization; not a performance result.
Attempt2 passed all three cold-cache measurements and validation. TTFT 5.875743 / 4.019455 / 3.994945 s; TPOT 781.923 / 751.909 / 742.240 ms; E2E 55.136902 / 51.389743 / 50.756050 s.
Full spread exceeds 5%; retain all samples, mark unstable and do not use a selected subset as a stable primary result. A separate bounded stability confirmation remains required.
Warmup and target prompts remain disjoint. No diagnostics or compilation inside measurements.

Across all three repeats, every rank produced identical generated-token hashes, final logical cache-state hashes, controller counters and peak allocated bytes. This rules out a changed logical workload/cache outcome in these samples, but does not identify the timing-variation cause.
