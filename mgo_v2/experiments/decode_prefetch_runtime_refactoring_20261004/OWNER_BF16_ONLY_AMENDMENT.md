# Owner amendment: BF16 only

The owner requested BF16 and no further time spent establishing numerical precision certainty. This supersedes FP32 main-stack selection and the additional BF16 candidate comparison in prior amendments/configuration.

Use BF16 partial accumulation and return for all subsequent BR tuning and BR/LA runtime arms. Do not run FP32/FP64 comparisons or separate legacy numerical-reference passes. Preserve previously collected numerical evidence; do not claim exact parity or free-generation quality. Ordinary compilation warmup, finite checks, frozen input identity, cache/role/copy/transport accounting remain. Timing jitter repeats remain independently authorized.

The FP32 M13_COALESCED_SCREEN queue is intentionally stopped after its current completed sample. Its samples remain archived and cannot be mixed into BF16 timing results. Restart the BR-only screen as M13_BF16_SCREEN with a fresh receipt.
