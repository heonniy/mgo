# R8 serial GEMM + inline H2D: quota comparison with Near placement in every arm

Same serial setup as `../serial_inline_r8_b16_20261011`. Stage 2 is Near in all arms. Only the
per-rank miss quota differs:
- NEAR (`LA_CA_NEAR`): balanced (max-min <= 1), remainder to low rank ids (the slow GPUs 0-3).
- FAST (`NEAR_FAST`): balanced, remainder to the fast GPUs 4-7.
- RANDOM_QUOTA (`RANDOM_QUOTA_NEAR`): unbalanced. Each miss draws a uniform random rank (seeded
  per event) and the counts become the quota, with Near placement inside it. At ~70 misses per layer
  the busiest rank gets ~1.4x the mean.

Two rounds with rotated order (N,F,R / R,F,N), 2 target repeats per job (4 samples per arm).
