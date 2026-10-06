# Partial primary checkpoint

Global64/input256/output64,4ranks on GPUs0,1,4,5. Warmup and first primary PASS; repeats2/3 pending. Primary1 TTFT6.091647906s, TPOT4.825709888s, E2E310.111370823s. Reconstructed outer timing from common release/max first/max end matches exactly. All manifest IDs covered once,64 output tokens/request, finite logits, GPU KV, cold parameter state and live residency cap checks pass. No final median/stability or cross-system gain claim.

Raw job: /home/hwlee/mgo-results/headline_r4_20261007/deepspeed_B16_L256_primary1
