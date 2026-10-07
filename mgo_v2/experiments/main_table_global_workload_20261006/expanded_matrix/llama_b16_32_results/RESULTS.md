# R4 C30 B16 L256 synchronous llama CPU32 results

Explicit32/32 threads, fixed32-CPU affinity, global64 requests, output64.
Placement and step-time recomputation PASS; two primaries after disjoint warmup.
No repeat selection or extra repetitions. Seconds, sample standard deviation.

|Repeat|TTFT|TPOT|E2E|
|---|---|---|---|
|1|71.758639|0.446404|99.882099|
|2|74.291851|0.480085|104.537231|

|Metric|Mean ± SD|Relative range|
|---|---|---|
|TTFT|73.025245 ± 1.791251|3.47%|
|TPOT|0.463245 ± 0.023816|7.27%|
|E2E|102.209665 ± 3.291675|4.55%|

TPOT range exceeds5%; do not label timing stable. Earlier main-table llama used32/64 threads without this fixed affinity, and its table selected3 of5. Changes include both thread budget and affinity, so this is not a thread-count-only causal comparison.
No claim of full16/32/64 token parity; those other conditions were canceled.
Original main-table results remain unchanged. No additional GPU experiments queued.
