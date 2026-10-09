# D2: full-resident, no-demand-copy reference

This is the plan's **static full-resident reference**, not the optional
schedule-preserving no-copy oracle. The same DeepSeek ShareGPT B16/input512,
fixed-continuation/frozen-router trace ran twice on R4 GPUs 0/1/4/5. All
1,664 routed experts were loaded to HBM before each measured generation with
the deterministic `global_expert_id mod 4` owner map, 416 expert slots/rank
plus two reserved physical slots. Every routed layer asserted zero demand
fetches, and all ranks recorded **zero measured H2D bytes**.

| Reference | TPOT (ms/token) | TTFT (s) | E2E (s) | Decode expert groups | Decode native waves |
|---|---:|---:|---:|---:|---:|
| Repeat 1 | 241.122 | 0.851 | 16.042 | 102,818 | 6,552 |
| Repeat 2 | 244.080 | 0.853 | 16.230 | 102,818 | 6,552 |

Mean TPOT was **242.601 ms/token**, with a 1.22% relative two-repeat range;
under the repetition rule, no third run was added. The two repeats produced
the same 64 complete output sequences. Maximum measured HBM allocation was
12.30 GiB/rank, consistent with the [preflight](D2_PREFLIGHT.md) and far
below the owner GPU limit. The guarded supervisor restored all four owner
model loads after each run.

The fixed-route C50 Near median was **279.949 ms/token**. The static fully
resident reference averaged 37.348 ms/token faster, but it also used a
different owner map and communication pattern: peer dispatch increased from
0.983 to 1.015 GiB over decode, and the busiest rank's expert-group count
fell from about 26,284 to 25,758. Native waves collapsed from roughly
13,365–13,640 per batch at C50 to exactly 6,552, one wave per
26 layers × 63 decode intervals × 4 ranks. The output sequences also differ
from C50 even though the fed tokens and router tensors were fixed.

This reference shows that substantial speed remains available in an
all-resident execution regime. It **cannot** attribute the 37 ms difference
to PCIe copies alone or bound the speedup of a feasible C50 policy: HBM
budget, owner placement, load balance and wave grouping all changed.
The D3 layer-level diagnostic will distinguish their visible timing effects
as far as the current measurement hooks allow. Aggregate checks are in
`D2_FULL_RESIDENT.json`; raw tokens and logs remain outside Git.
