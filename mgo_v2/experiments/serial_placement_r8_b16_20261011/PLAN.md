# R8 serial GEMM + inline H2D + FAST quota: Near vs worst stage-2 placement

Same serial setup as `../serial_inline_r8_b16_20261011`: `--ours-mode A --h2d-serial-ablation
--inline-demand-h2d`, prefetch OFF, frozen Near teacher tokens. Both arms use the FAST quota
rows, so they differ only in stage 2 (which miss goes to which rank).

- FAST+Near (`NEAR_FAST`): baseline for this comparison.
- FAST+Worst (`FAST_WORST`): mirror of Near. Maximizes projected critical-rank expert rows
  and, within 2%, minimizes rank-local demand. CPU check on the ShareGPT proxy gave
  critical/mean rows 2.82 vs 1.14 and a remote fraction of 0.90 vs 0.81.

Three rounds with alternating order (N,W / W,N / N,W), 2 target repeats per job (6 samples per arm).
