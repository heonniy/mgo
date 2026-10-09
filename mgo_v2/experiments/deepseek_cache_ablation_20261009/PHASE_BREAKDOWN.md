# D3: where the fixed-route DeepSeek decode time goes

The same D1 fixed continuation and frozen router were profiled on C20 and
C50. Each guarded job generated all 64 tokens but recorded detailed data
only for the **first eight decode steps × 26 routed layers = 208 layer events**
per rank. Both jobs matched their clean-capacity output tokens and full
decode H2D exactly. Instrumented TPOT was 289.888/280.837 ms/token, just
0.62%/0.32% above the corresponding clean median; it is diagnostic, not a
replacement primary sample.

The table gives the **median over 208 layer events of that event's maximum
rank value**, in ms per routed layer. The rank providing the maximum can
differ by phase. CUDA intervals include host submission gaps and collective
peer waits; H2D runs on another stream. These columns must not be added to
reconstruct TPOT.

| Measured layer interval | C20 | C50 |
|---|---:|---:|
| Metadata host path, including all-gather/D2H | 1.398 | 1.719 |
| Of metadata: receive-to-host wait | 0.918 | 1.251 |
| Placement controller host work | 0.085 | 0.085 |
| Rank-partial layout and slot binding host work | 0.284 | 0.287 |
| Demand-H2D enqueue host work | 0.271 | 0.154 |
| Dense dispatch-payload submission host work | 0.865 | 0.848 |
| Dispatch CUDA-stream interval | 1.219 | 1.053 |
| Native expert CUDA-stream interval | 3.558 | 3.470 |
| Return collective **plus source combine** CUDA interval | 2.974 | 3.249 |
| First dispatch to return/combine completion | 7.479 | 7.246 |
| Explicit main-stream H2D wait | 0.000 | 0.000 |
| Native-wave host submission | 3.364 | 3.333 |

Across the sampled layers, demand fetches fell **10,386→6,413** and native
ready waves fell **3,015→1,675**. Yet the median expert interval improved by
only 0.088 ms/layer and native-wave host submission by 0.031 ms/layer.
The number of executed expert groups was unchanged under fixed routing:
fewer copies and fewer waves do not remove the separate gate/up/down GEMMs
for each expert. The controller itself was only about 0.085 ms/layer in both
capacities. A standalone expert probe measured one 16.5-MiB H2D copy at
0.391 ms and three small-row GEMMs at about 0.045 ms, but those isolated
services cannot be multiplied into an exposed TPOT because ready-first
execution overlaps the streams.

The fixed-route C50 placement also shows possible **offsetting costs**.
Median host return-submission skew across four ranks rose from 0.618 to
0.863 ms/layer. The maximum/minimum rank expert-token-row ratio rose from
1.021 to 1.053 (p95 1.065→1.264). Return-plus-combine and metadata
receive-to-host intervals became longer at C50, even though their payload
formats did not change. The metadata D2H wait includes completion of the
preceding NCCL all-gather, so it is not pure memcpy time. The return
interval includes peer wait and `index_add_` combine, so it is not pure link
time. These observations are consistent with owner/rank timing offsetting
some H2D savings, but they do not split transport service from late-arrival
or combine work exactly.

The strongest supported explanation for the shallow C20→C50 TPOT curve is
that **fetches are real and expensive, but much of their service is hidden
behind ready expert work**. The exposed decode path still executes the same
expert groups and substantial return/metadata work. A larger cache changes
owners and can increase rank-row and arrival skew. The exact share of the
remaining 8.160-ms/token improvement attributable to copies, wave grouping
and owner placement is unresolved by this observational profile; a focused
same-capacity overlap ablation is next.

`LAYER_CRITICAL_PATH.csv` retains 416 aggregated layer rows and
`EXECUTOR_WAVES.csv` retains 64 rank-step aggregates. `D3_VALIDATION.json`
holds median/p95 values, route hashes, output/H2D parity and profiling
overhead. Raw per-rank event records stay in the guarded job outputs.
