# R8 serial GEMM + inline H2D + FAST quota: Near vs random stage-2 placement

Same setup as `../serial_placement_r8_b16_20261011`. Both arms use the FAST quota rows and differ only in
stage 2:
- FAST+Near (`NEAR_FAST`): baseline.
- FAST+Random (`FAST_RANDOM`): uniform random rank within the quota, seeded per event.
  CPU check on the ShareGPT proxy: critical/mean rows 1.53 vs 1.14 for Near.

Three rounds with alternating order, 2 target repeats per job (6 samples per arm).
