# Rank-level evidence (BR C60)

R4 ranks 0/1/2/3 map to GPUs 0/1/4/5. Batch is per rank: global batch is four times the label. Every time below is from the separate diagnostic, except explicitly marked primary counters. Decode totals are divided by 256, and include all 48 layers per step. Expert span includes CPU submission/gather work and GPU completion; it is not isolated GEMM time. Return span includes peer waiting. Phase columns are selected exclusive portions, not a complete partition.

## B8

|Prefetch|Rank / GPU|Expert groups/step|Token-expert rows/step|Demand misses/step|Expert span ms/step|Expert thread CPU ms/step|Return collective ms/step|Metadata ms/step|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|off|0 / 0|1025.8|3143.0|243.3|344.87|344.75|56.48|56.90|
|off|1 / 1|1013.8|3170.3|231.2|344.95|343.06|56.68|59.07|
|off|2 / 4|986.1|2999.6|218.6|330.65|329.66|70.59|63.37|
|off|3 / 5|980.1|2975.1|207.1|318.06|317.92|88.69|72.76|
|on|0 / 0|1023.8|3075.3|175.0|362.79|359.93|63.22|62.25|
|on|1 / 1|1019.1|3160.6|163.0|365.78|356.98|59.35|60.57|
|on|2 / 4|1002.4|3083.3|151.1|358.06|352.05|67.00|64.32|
|on|3 / 5|985.4|2968.7|139.1|339.86|335.18|91.86|76.23|

|Prefetch|Dispatch entry skew mean ms/layer|Expert entry skew mean ms/layer|Return entry skew mean ms/layer|Expert span spread p50 / p90 ms/layer|Slowest = most experts % (ties allowed)|Slowest = most token rows % (ties allowed)|
|---|---:|---:|---:|---:|---:|---:|
|off|0.672|0.266|2.550|2.129 / 3.467|95.1|60.5|
|on|0.847|0.282|2.636|2.206 / 3.745|95.1|59.1|

Entry skew is max minus min host API-entry timestamp across four ranks for the same layer/step, not GPU network start time. Matching the heaviest rank is observational; ties are allowed and become common at large batches.

|Prefetch|Issued|Useful|Wasted|Useful / issued %|Mandatory fetches|Promotion-victim reloads|Primary ready-first waits (sum ranks)|
|---|---:|---:|---:|---:|---:|---:|---:|
|off|0|0|0|N/A|236508|0|0|
|on|96256|72272|23984|75.08|166844|61254|0|

Controller counters are global primary counters including prefill; the speculative mechanism is used during decode. Useful is consumed prefetch, not a net saved-copy count. Promotion-victim reloads are events, not proof that all are additional misses caused by prefetch. Ready-first waits count calls taking an explicit not-ready path; zero does not mean H2D takes zero time.

## Interpretation boundaries

OFF and ON use identical MAIN and reserved capacities, streaming, ready-first and T2 synchronization. Only speculative prefetch submission is disabled in OFF. Native generated tokens differ across arms, so subsequent routing can differ. Detailed diagnostics reproduce their own primary tokens/cache/bytes. No frozen-route counterfactual or repeated timing confirmation is claimed.

H2D service distributions in SUMMARY.json include prefill and decode; copy-stream times overlap other work. Demand-H2D submission time is not DMA duration. Do not add service time to TPOT or subtract diagnostic MoE time from clean TPOT.

Placement recommendations must distinguish per-expert executor overhead, token-row work, exposed fetch dependency, and peer waiting. BR is the only measured placement; these observations cannot establish a measured LA/CA/FCA/Near winner.
