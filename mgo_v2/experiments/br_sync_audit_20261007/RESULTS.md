# BR current-runtime audit results

R4/C30, local B16 or B64 (global64/256), input256, output64, BR, H0/full-pinned, optimized prefill, original decode. One clean primary per cell; no stability estimate.

|Local batch|TTFT s|TPOT s|E2E s|
|---|---:|---:|---:|
|16|3.271677|0.785293|52.745124|
|64|4.659429|1.021785|69.031866|

## Decode cache reuse

Expert-use denominators deduplicate each expert within a layer/step/rank. Token-weighted denominators count every routed token-expert use. These are logical hits avoiding a new demand copy, not a claim of zero physical H2D wait.

|Batch / window|MAIN hit|Prefetch hit|Avoid demand|Token-weighted avoid demand|Surviving prefill hit|
|---|---:|---:|---:|---:|---:|
|B16 / first_step|41.42%|12.07%|53.49%|75.55%|41.42%|
|B16 / all_decode|38.96%|8.36%|47.32%|76.18%|13.26%|
|B64 / first_step|26.96%|9.60%|36.56%|58.38%|26.96%|
|B64 / all_decode|31.29%|6.86%|38.15%|75.97%|2.45%|

## Separate diagnostic: per-rank mean milliseconds per decode token

Ranges below are minimum–maximum across four ranks. These completion spans include CPU submission gaps and peer waits; expert_compute is not isolated GPU GEMM time. H2D service overlaps and is non-additive. Diagnostic overhead can also change ready-first timing; do not substitute these spans for primary TPOT.

|Phase|B16 ms/token|B64 ms/token|
|---|---:|---:|
|Metadata|51.149–71.828|49.020–89.970|
|Placement controller|15.328–15.952|19.505–20.420|
|CPU layout|28.912–29.971|48.434–52.312|
|GPU index materialization|32.939–34.539|60.534–64.083|
|Dispatch submission + completion|75.902–76.766|106.105–114.277|
|Required H2D exposed dependency|0.000–0.000|0.000–0.000|
|Expert execution incl. host launches|392.839–423.007|497.018–572.484|
|Return collective|36.644–74.846|36.288–116.872|
|Local partial / combine|24.847–28.025|28.226–33.773|
|Prefetch controller|34.941–37.956|35.549–37.931|

See STRUCTURE.md for exact barrier and stream dependencies. Prefill has a required-H2D global barrier; overlapped decode does not. Neither selected path adds a post-expert global barrier. Collective completion can include slower-peer waiting.

All diagnostic tokens and final cache/role hashes match the corresponding primary. Primary receipts report no compilation. Full events remain in the raw paths recorded in SUMMARY.json; all requested measurements are preserved.

B16 launched at fb30aab; the diagnostic module was extended during warmup before its first lazy import (675896e and 3daa20a). Runtime/primary semantics were unchanged. SOURCE_HASHES.json records the executed diagnostic implementation. No additional run or optimization is authorized by this audit.

## Findings and limits

1. B64 return completion is rank-skew sensitive: GPU0 expert span572.48ms/token and return36.29ms, GPU1 expert497.02ms and return116.87ms. This is consistent with faster ranks waiting for the slower peer; it does not isolate pure network time or prove which upstream subcomponent causes all skew.
2. Expert spans contain substantial host work: own-thread CPU time is377.59–412.87ms/token for B16 and488.10–538.81ms/token for B64. H0 launches experts individually; host readiness polling, gathers, launches and weighting remain. CPU layout plus GPU-index preparation also grows from about62–65ms to109–116ms/token. These are diagnostic costs, not a measured speedup opportunity of the same size.
3. No per-expert required-H2D waits were observed during diagnostic decode. Copies still total1547.98GiB (B16) and2138.40GiB (B64) across prefill+decode, all ranks, including prefetch. Their service overlaps the critical stream. Diagnostic overhead gives transfers more time to finish, so this does not prove zero H2D impact in uninstrumented primary timing.
4. H2D medians span0.181–0.200ms/copy in B16 and0.184–0.211ms in B64; p99 reaches0.765ms and0.711ms respectively. These samples show tails and rank variation but do not establish severe contention as the dominant bottleneck; no interference-controlled ablation was run.
5. MAIN hits persist, but original prefill entries gradually turn over. Continuously surviving prefill entries account for13.26% of all decode expert uses in B16 and2.45% in B64. Total token-weighted no-new-demand coverage remains about76% in both. Prefetch hits avoid a new demand request but consumed earlier H2D; no-prefetch causal savings were not measured.
6. Instrumented decode averages about0.857s/token and1.108s/token versus clean0.785s and1.022s. Diagnostic overhead is observable; never mix these timings. One primary per condition does not establish stability or a BR-versus-Near policy gain.
7. Existing main-table OURS already used P2/T2 prefetch. A historical B16/L256 primary has23257 useful promotions. This audit changes only policy to BR and enables separate post-primary diagnostics; the default policy remains Near and the original decode remains default.
