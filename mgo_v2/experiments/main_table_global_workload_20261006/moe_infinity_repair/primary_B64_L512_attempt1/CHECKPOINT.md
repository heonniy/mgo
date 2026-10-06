# Partial large-cell primary checkpoint

B64/rank label means global256 requests, input512, output64. Attempt1 completed warmup and primary repeat1. Median and stability await repeats2/3.

Repeat1: TTFT9.722507175s, TPOT3.625780713s, E2E238.146692107s. All256 requests generated64 tokens, expert residency starts empty, finite GPU attention/KV checks pass, peak expert charged bytes17392730112 respects the limit. No final cross-system claim.

Raw job: /home/hwlee/mgo-results/headline_r4_20261007/infinity_B64_L512_primary1

**SUPERSEDED for headline use:** the harness retained the previous batch DynamicCache in the local `kv` variable during the next generation. No prefix reuse occurred, but live HBM was inflated (about3.37GiB/GPU for global256). A corrected harness explicitly releases the cache and verifies weak references before the next batch. Original receipts remain unchanged; they are not eligible primary results.
