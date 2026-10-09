# D1: DeepSeek C20/C50 with fixed continuation and router trace

Six guarded R4 runs passed on GPUs 0/1/4/5 in the order
**C20, C50, C20, C50, C20, C50**. Each run used the same ShareGPT 64-request
batch, input512/output64, Near/native main_OURS, prefetch OFF, a separate
warmup, and an empty expert cache at target start. The next-token inputs,
selected expert IDs, BF16 router weights and full router probabilities were
loaded from a single C20 reference. The route-capture run reproduced all
64 original C20 output sequences before replay; its timing is excluded.

| Cache | Clean TPOT repeats (ms/token) | Median [full range] | Decode MAIN hit, rank-local distinct groups | Decode demand H2D, four ranks |
|---:|---:|---:|---:|---:|
| C20 | 287.503, 288.559, 288.109 | 288.109 [287.503, 288.559] | 15.65% | 1,397.489 GiB |
| C50 | 279.644, 284.856, 279.949 | 279.949 [279.644, 284.856] | 47.03% | 877.610 GiB |

The **median TPOT reduction is 8.160 ms/token, or 2.83%**, under fixed
routing. Every matched pair favored C50 by 7.859, 3.703 and 8.160 ms/token.
The second C50 repetition is visibly higher than its first and third;
neither it nor any other result was removed. C20's range is 1.056 ms/token;
C50's is 5.212 ms/token. TTFT and E2E samples, raw source commits and
per-rank counters are retained in `MATCHED_ROUTE_VALIDATION.json`.

Both capacities executed **102,818** rank-local distinct-expert groups over
decode. Since each DeepSeek expert copy is exactly 16.5 MiB and prefetch is
OFF, the H2D counters correspond to 86,729 C20 and 54,465 C50 demand
copies. The resulting MAIN hit fractions are 15.65% and 47.03%; H2D bytes
fell **37.2%**. The ready-first native executor made 27,725–30,779 waves
at C20 and 13,365–13,640 at C50. It made the same number of expert groups,
but C50 shifted their owners slightly: decode peer dispatch rose from 0.970
to 0.983 GiB, and peer return from 0.965 to 0.979 GiB. Those volumes are
not a measurement of collective service time.

All three runs within each capacity produced the same 64 complete output
sequences. Only **35/64** complete sequences matched between C20 and C50,
despite the frozen router trace. Different BF16 summation/owner order can
change logits, but the recorded next-token inputs and all injected routing
tensors were identical; future demand was therefore controlled. This is a
valid *fixed-route capacity/placement* comparison, not a copy-only
intervention. Cache capacity also changes owner assignment and packet
distribution, so the 8.160 ms figure cannot be assigned entirely to DMA.

The result confirms that a large logical hit/H2D improvement produces only
a small TPOT improvement **even after future routing is held fixed**. It
does not prove that H2D is irrelevant: the normal executor can overlap
copies and ready expert work. The older explicit-wait diagnostic found at
most 0.327 ms/token of direct main-stream H2D wait; that narrow counter
does not include indirect PCIe/HBM contention or wave fragmentation. The
standalone 16.5-MiB copy measured 0.391 ms, versus about 0.045 ms for
three small-row GEMMs, but these operations overlap in the full runtime.
The next phase must measure the layer critical path before naming the
remaining dominant component or changing placement.

`MATCHED_ROUTE_VALIDATION.json` holds the aggregate checks and SHA-256
references. Full token arrays, route tensors, logs and resource receipts
remain outside Git under the guarded result directory.
