# R8 serial GEMM + inline H2D: quota comparison (Near placement in every arm)

Owner-requested scope cut: only the random quota needed new samples, because NEAR and FAST already have
serial-mode samples in `../serial_inline_r8_b16_20261011` and the FAST+Near rows of the two serial
placement runs. Round 1 ran all three arms. Round 2 ran only RANDOM_QUOTA, then the runner stopped on
the STOP file before the remaining FAST and NEAR jobs. STATUS.json therefore says FAIL (AssertionError
on STOP). Every completed job passed.

| Arm | n | TPOT median [range], s | vs NEAR (round 1) |
|---|---:|---:|---:|
| NEAR (balanced, remainder to GPUs 0-3) | 2 | 0.4298 [0.4262, 0.4333] | 0 |
| FAST (balanced, remainder to GPUs 4-7) | 2 | 0.4274 [0.4242, 0.4307] | -0.55% |
| RANDOM_QUOTA (unbalanced, i.i.d. rank per miss) | 4 | 0.4791 [0.4763, 0.4831] | **+11.4%** |

Consistency with earlier serial runs: NEAR 0.4299 median (6 samples, serial_inline); FAST+Near
0.4247-0.4278 across three runs. Round 2 RANDOM_QUOTA (0.4831 / 0.4771) matches round 1 (0.4812 / 0.4763).

- **Quota balance is worth ~11%.** NEAR and FAST differ only in which ranks get the +1, and that
  gap is small (-0.5%). Dropping the balance rule entirely costs +11.4%.
- **Run totals hide the per-layer imbalance.** Over the whole run RANDOM_QUOTA's per-rank copies
  are the most even (26.4-26.9k vs 25.1-27.8k), and its summed max H2D wait is the lowest
  (3.67 s vs 4.46 s). Each layer still has a busiest rank with ~1.4x the mean misses, both copies and
  serial expert compute, and all ranks wait for it every layer.
- Total fetches are +0.9% (213,218 vs 211,249), a small cache side effect.
