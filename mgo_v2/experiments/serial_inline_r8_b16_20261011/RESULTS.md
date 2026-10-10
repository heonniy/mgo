# R8 serial GEMM + inline H2D: NEAR vs FAST

Three rounds with alternating order, 2 target repeats per job (6 samples per arm). Full table: `RESULTS_TABLE.md`.

| Arm | TPOT median [range], s | vs NEAR | per round | max H2D wait, GPUs 0-3 / 4-7 |
|---|---:|---:|---|---|
| NEAR | 0.4299 [0.4268, 0.4355] | 0 | — | 4.41 / 3.71 s |
| FAST | 0.4278 [0.4210, 0.4363] | -0.50% | -1.36, -0.30, +0.50 | 4.09 / 3.93 s |

- **No measurable FAST gain in serial mode.** Rounds disagree in sign and the ranges overlap.
- FAST does rebalance H2D: the largest per-rank wait drops by 0.32 s of ~27 s decode, and the
  4-7 wait rises from 3.71 to 3.93 s. If that H2D saving were fully critical it would be ~1.2%.
  It mostly is not, because serial per-expert compute (~0.2 ms each) dominates each rank's time.
  FAST also moves the extra misses' compute onto GPUs 4-7.
- **Repeat-position effect:** in every job, target repeat 1 is ~1.5% slower than repeat 2
  (NEAR 0.4342/0.4273, 0.4355/0.4286, 0.4313/0.4268; FAST 0.4288/0.4210, 0.4348/0.4267,
  0.4363/0.4261). Pairing by repeat position gives FAST +0.1% (repeat 1) and -0.3% (repeat 2).
  The one-target warmup is not enough in serial mode.
