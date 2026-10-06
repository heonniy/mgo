# Stock DeepSpeed ZeRO-3 CPU-offload validation

Smoke1 PASS on GPUs0,1,4,5: calibration, warmup, cold target. Global4/input32/output2 is validation only, never headline timing. All4 ranks report expert peak2925527040 bytes, all-parameter peak3115216896 bytes <=4348182528-byte per-rank cap, and pinned CPU shards15266061312 bytes/rank. Cache reset proves all parameters NOT_AVAILABLE before each batch. No custom DeepSpeed runtime patch. Main cells remain pending.

Before primary execution, added an O(1) read of the stock coordinator live-parameter counter at fetch boundaries. Calibration checks it against the full parameter/status scan; primary repeats assert the same byte cap and report actual maxima. GPU KV-residency is checked after generation. This observes the stock runtime without altering its prefetch/release policy; the new checks will be exercised by the next full job.
