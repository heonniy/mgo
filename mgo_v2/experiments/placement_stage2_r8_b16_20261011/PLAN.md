# R8 grouped (inline H2D): stage-2 placement under a fixed FAST quota

Same cell and protocol as `../haq_fast_r8_b16_20261011`: inline demand H2D, grouped
hit-then-miss, prefetch OFF, frozen Near teacher tokens. Stage 1 (per-rank miss
quota) is the FAST table for the three FAST arms. Only stage 2 (which miss goes to
which rank inside the quota) changes.

- NEAR (`LA_CA_NEAR`): original quota and Near placement. Overall baseline.
- FAST+Near (`NEAR_FAST`): Near placement. Minimizes projected critical-rank expert
  rows and, within 2%, maximizes rank-local demand.
- FAST+Random (`FAST_RANDOM`): uniform random rank within the quota, seeded per event.
- FAST+Worst (`FAST_WORST`): mirror of Near. Maximizes projected critical-rank rows
  (worst compute imbalance) and, within 2%, minimizes rank-local demand (worst comm).

CPU check (`cpu_check.py`, ShareGPT pool proxy, 32 steps), critical/mean expert rows and remote row fraction:
Near 1.144 / 0.812, Random 1.531 / 0.874, Worst 2.820 / 0.903.

Two rounds with rotated order, 2 target repeats per job (4 samples per arm).
