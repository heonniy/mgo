# D4: same-capacity H2D overlap intervention

DeepSeek-V2-Lite, frozen ShareGPT R4/B16/input512/output64, GPUs 0/1/4/5, Near/native, prefetch OFF. Two target repeats per arm follow an independent warmup and cold cache in balanced normal–serial–serial–normal order. The serial arm waits for current-layer demand H2D after dispatch before expert execution; normal retains ready-first overlap. All guarded jobs passed and owner inference loads were restored.

| Cache | Normal TPOT, ms/token | Serial TPOT, ms/token | Serial penalty | Decode demand H2D |
|---:|---:|---:|---:|---:|
| C20 | 289.253 [284.971, 293.536] | 352.994 [352.684, 353.303] | +63.740 ms/token (22.0%) | 1397.489 GiB |
| C50 | 285.946 [285.505, 286.386] | 316.603 [314.982, 318.224] | +30.657 ms/token (10.7%) | 877.610 GiB |

Both arms at each capacity used identical frozen request IDs, fed tokens, router SHA, 102,818 executed expert groups, actual decode H2D bytes and cache capacities. All 64 complete output sequences matched between both arms and repetitions at each capacity. No measured serial TPOT overlaps the normal full range. C20 normal repeats differed by 2.96% of their mean, so its full range is shown; the separation remains at least 59.148 ms/token using the slowest normal and fastest serial samples. C50 normal repeats differed by 0.31%.

The serial intervention also reduced ready-wave counts to 6,552 at either capacity: C20 normal waves were 26,943 and 32,023; C50 normal waves were 13,533 and 13,381. Thus the TPOT penalty measures the net change from disabling ready-first overlap **and** changing wave grouping. It does not isolate PCIe copy service or directly predict a faster cache. The C20→C50 H2D reduction is real, while D1 shows only an 8.160-ms/token median TPOT improvement: most copy service is hidden in the normal executor, and unchanged expert work plus return/metadata and changed rank ownership limit the visible gain. D3 measured the controller at about 0.085 ms/routed layer and found return/metadata and rank skew increased at C50. A pure transport-versus-peer-wait split remains unmeasured.

Raw target and rank records are outside Git at `/home/hwlee/mgo-results/headline_r4_20261007/dca_d4_overlap_c{20,50}_ns_sn_v1/`. `D4_OVERLAP.json` retains the exact source revision, two-repeat ranges, counters and parity checks.
