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

|Prefetch|Rank|Primary decode peer GiB|Primary decode H2D GiB|Decode H2D service p50 / p90 / p99 ms per copy|
|---|---:|---:|---:|---:|
|off|0|2.055|547.365|0.190 / 0.227 / 0.253|
|off|1|2.045|520.233|0.214 / 0.264 / 0.371|
|off|2|2.037|491.889|0.187 / 0.249 / 0.354|
|off|3|2.043|466.075|0.184 / 0.227 / 0.304|
|on|0|2.049|605.276|0.193 / 0.236 / 0.413|
|on|1|2.045|578.153|0.231 / 0.304 / 1.053|
|on|2|2.043|551.408|0.216 / 0.293 / 0.417|
|on|3|2.039|524.443|0.213 / 0.275 / 0.413|

Decode H2D percentiles exclude the initial copies using the recorded prefill byte boundary; their byte sum is checked against primary decode H2D bytes. They measure copy service, not queue delay or exposed critical-path wait.

## B16

|Prefetch|Rank / GPU|Expert groups/step|Token-expert rows/step|Demand misses/step|Expert span ms/step|Expert thread CPU ms/step|Return collective ms/step|Metadata ms/step|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|off|0 / 0|1229.8|5915.7|379.3|444.43|429.50|45.79|55.06|
|off|1 / 1|1224.6|6330.3|367.1|407.57|407.06|88.11|76.64|
|off|2 / 4|1214.7|6291.1|355.4|401.86|402.05|94.37|81.13|
|off|3 / 5|1199.5|6038.8|343.5|417.71|405.77|81.42|73.49|
|on|0 / 0|1226.9|5952.4|291.3|426.66|419.79|59.76|63.18|
|on|1 / 1|1223.4|6285.1|279.5|429.95|418.69|61.73|65.31|
|on|2 / 4|1211.7|6261.2|267.4|421.23|412.68|70.85|69.64|
|on|3 / 5|1196.3|6077.3|255.3|409.31|401.26|86.18|74.18|

|Prefetch|Dispatch entry skew mean ms/layer|Expert entry skew mean ms/layer|Return entry skew mean ms/layer|Expert span spread p50 / p90 ms/layer|Slowest = most experts % (ties allowed)|Slowest = most token rows % (ties allowed)|
|---|---:|---:|---:|---:|---:|---:|
|off|1.094|0.340|2.957|2.471 / 4.251|83.0|53.5|
|on|1.023|0.347|2.669|2.204 / 3.914|93.7|58.4|

Entry skew is max minus min host API-entry timestamp across four ranks for the same layer/step, not GPU network start time. Matching the heaviest rank is observational; ties are allowed and become common at large batches.

|Prefetch|Issued|Useful|Wasted|Useful / issued %|Mandatory fetches|Promotion-victim reloads|Primary ready-first waits (sum ranks)|
|---|---:|---:|---:|---:|---:|---:|---:|
|off|0|0|0|N/A|376056|0|1|
|on|96256|89688|6568|93.18|285997|81878|0|

Controller counters are global primary counters including prefill; the speculative mechanism is used during decode. Useful is consumed prefetch, not a net saved-copy count. Promotion-victim reloads are events, not proof that all are additional misses caused by prefetch. Ready-first waits count calls taking an explicit not-ready path; zero does not mean H2D takes zero time.

|Prefetch|Rank|Primary decode peer GiB|Primary decode H2D GiB|Decode H2D service p50 / p90 / p99 ms per copy|
|---|---:|---:|---:|---:|
|off|0|4.064|853.427|0.229 / 0.298 / 0.391|
|off|1|4.097|826.014|0.188 / 0.217 / 0.274|
|off|2|4.087|799.646|0.180 / 0.214 / 0.231|
|off|3|4.072|772.778|0.224 / 0.298 / 0.385|
|on|0|4.079|866.927|0.201 / 0.272 / 1.016|
|on|1|4.095|840.445|0.225 / 0.302 / 1.120|
|on|2|4.082|813.067|0.214 / 0.287 / 1.108|
|on|3|4.087|785.892|0.214 / 0.285 / 1.066|

Decode H2D percentiles exclude the initial copies using the recorded prefill byte boundary; their byte sum is checked against primary decode H2D bytes. They measure copy service, not queue delay or exposed critical-path wait.

## Interpretation boundaries

OFF and ON use identical MAIN and reserved capacities, streaming, ready-first and T2 synchronization. Only speculative prefetch submission is disabled in OFF. Native generated tokens differ across arms, so subsequent routing can differ. Detailed diagnostics reproduce their own primary tokens/cache/bytes. No frozen-route counterfactual or repeated timing confirmation is claimed.

Unprefixed H2D service fields in SUMMARY.json include prefill; decode-prefixed fields exclude it. Copy-stream times overlap other work. Demand-H2D submission time is not DMA duration. Do not add service time to TPOT or subtract diagnostic MoE time from clean TPOT.

Placement recommendations must distinguish per-expert executor overhead, token-row work, exposed fetch dependency, and peer waiting. BR is the only measured placement; these observations cannot establish a measured LA/CA/FCA/Near winner.
