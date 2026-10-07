# C60 MAIN eviction replay

Same frozen C30 source traces; CPU counterfactual only. R4, input256,256 decode steps after prefill. Prefetch OFF; all3686 slots MAIN. Counts exclude prefill.

|Batch|Policy|C30 hit %|C60 hit %|C60 hit|Miss|Eviction|Reload|
|---|---|---:|---:|---:|---:|---:|---:|
|8|gate-score|42.41|77.51|793755|230304|230304|230282|
|8|LFU-reset|0.00|71.89|736180|287879|287879|287857|
|8|LFU-cumulative|40.08|81.27|832230|191829|191829|191807|
|8|LRU-reset|0.00|9.34|95691|928368|928368|928346|
|8|LRU-cumulative|0.00|9.34|95691|928368|928368|928346|
|16|gate-score|35.34|69.97|876699|376289|376289|376273|
|16|LFU-reset|0.00|51.89|650154|602834|602834|602818|
|16|LFU-cumulative|16.65|71.87|900473|352515|352515|352499|
|16|LRU-reset|0.00|0.47|5864|1247124|1247124|1247108|
|16|LRU-cumulative|0.00|0.47|5864|1247124|1247124|1247108|
|64|gate-score|29.95|61.89|914066|562745|562745|562739|
|64|LFU-reset|0.00|0.12|1790|1475021|1475021|1475015|
|64|LFU-cumulative|0.00|25.20|372210|1104601|1104601|1104595|
|64|LRU-reset|0.00|0.00|7|1476804|1476804|1476798|
|64|LRU-cumulative|0.00|0.00|7|1476804|1476804|1476798|

All trace hashes match C30. All hit/miss, occupancy, eviction/reload conservation and LRU-equivalence checks pass. C60 does not claim physical Gate parity against the C30 cache state or physical TPOT gains. Reset/cumulative refers to eviction-time policy metadata; analytical reload history is always retained.

B64 LRU has7 hits (0.000474%), not exactly zero; table percentages are rounded.
