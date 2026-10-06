# Stock DeepSpeed ZeRO-3 CPU-offload validation

Smoke1 PASS on GPUs0,1,4,5: calibration, warmup, cold target. Global4/input32/output2 is validation only, never headline timing. All4 ranks report expert peak2925527040 bytes, all-parameter peak3115216896 bytes <=4348182528-byte per-rank cap, and pinned CPU shards15266061312 bytes/rank. Cache reset proves all parameters NOT_AVAILABLE before each batch. No custom DeepSpeed runtime patch. Main cells remain pending.
