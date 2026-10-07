# MAIN eviction history results

R4/C30, input256, empty cache before prefill; prefill retained into256 decode forwards. Prefetch OFF. Local batches8/16/64; global32/64/256. LFU counts one use per distinct layer/step expert. Counts below exclude prefill.

|Batch|Policy|MAIN hit|Miss|Hit %|Eviction|Reload|
|---|---|---:|---:|---:|---:|---:|
|8|gate-score|434276|589783|42.41|589783|589761|
|8|LFU-reset|0|1024059|0.00|1024059|1024037|
|8|LFU-cumulative|410394|613665|40.08|613665|613643|
|8|LRU-reset|0|1024059|0.00|1024059|1024037|
|8|LRU-cumulative|0|1024059|0.00|1024059|1024037|
|16|gate-score|442830|810158|35.34|810158|810142|
|16|LFU-reset|0|1252988|0.00|1252988|1252972|
|16|LFU-cumulative|208652|1044336|16.65|1044336|1044320|
|16|LRU-reset|0|1252988|0.00|1252988|1252972|
|16|LRU-cumulative|0|1252988|0.00|1252988|1252972|

Reset deletes policy frequency/recency on eviction; cumulative preserves history across eviction and re-admission. Analytical reload history is always retained and cannot affect replacement. A hit is an active expert already resident before admission, not a prefetch or token-weighted metric. Eviction counts actual removed residents; reload counts a miss for any previously admitted expert, including prefill residents.

Per-step CSVs include prefill at step0 and separate decode-only cumulative counters. Checkpoints1/8/16/32/64/128/256 are in each SUMMARY.json. LRU reset/cumulative equivalence and exact Gate capture/replay event counts plus final cache parity are required.

BR algorithm/seed42 and total1843 MAIN slots are fixed; owner maps may differ because policy-dependent misses alter admissions. These are replay cache metrics conditional on the same frozen source trace per batch, not physical timing or quality measurements. No-prefetch source capture is exact-only BF16 with gate eviction.
