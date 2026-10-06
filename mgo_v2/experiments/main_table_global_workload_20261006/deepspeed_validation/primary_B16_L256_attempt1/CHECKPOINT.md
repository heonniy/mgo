# Complete unbound-CPU primary attempt; stability gate not passed

Global64/input256/output64,4ranks on GPUs0,1,4,5. All3 primaries pass original manifest coverage, raw-clock timing, finite logits, GPU KV, cold parameter state and live residency cap checks.

TTFT samples6.091647906/5.416454479/5.461541401s; TPOT4.825709888/4.128344288/4.127035073s; E2E310.111370823/265.502144644/265.464750974s. First sample is slower; retain all3. The attempt is UNSTABLE and not selected for headline use.

A bounded confirmation with fixed per-rank CPU ranges is queued, without altering the stock DeepSpeed algorithm or workload. All GPUs belong to the same reported NUMA0; CPU placement observation does not prove NUMA contention or establish the variance cause.

Raw job: /home/hwlee/mgo-results/headline_r4_20261007/deepspeed_B16_L256_primary1
