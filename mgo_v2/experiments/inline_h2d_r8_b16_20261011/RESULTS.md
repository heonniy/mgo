# Inline one-pass demand H2D submission on R8 grouped decode

Qwen3-30B ShareGPT R8/C30/local-B16/input512/output64. Grouped hit-then-miss,
prefetch OFF, frozen Near teacher tokens. Two rounds with rotated order, two target
repeats per job (4 samples per arm). Baseline: NEAR with the staging-thread submission.
Full table: `RESULTS_TABLE.md` (`python ../haq_fast_r8_b16_20261011/summarize_arms.py . base_near`).

| Arm | TPOT median [range], s | vs NEAR | per round |
|---|---:|---:|---|
| NEAR, staging thread | 0.2886 [0.2868, 0.2906] | 0 | — |
| NEAR, inline | 0.2796 [0.2789, 0.2804] | **-3.10%** | -3.65, -2.60 |
| FAST, staging thread | 0.2840 [0.2795, 0.2848] | -1.59% | -1.97, -1.91 |
| FAST, inline | 0.2749 [0.2745, 0.2756] | **-4.74%** | -5.23, -4.24 |

- Inline submission and FAST each help, and their effects add up: inline is about -3.1 to -3.2%, FAST is
  about -1.6 to -1.7%, and both together give -4.7%. Each inline arm's full range lies below the matching
  staging-thread arm's range.
- Fetch counts are identical within each quota (NEAR 211,249; FAST 211,346).
- Do not compare the "max wait" columns across submission modes. With inline submission the
  host spends the copy-launch time before the measured wait begins, so the window starts at a
  different point.
- Provenance: `base_near_v1`, `inl_near_v1` and `base_fast_v1` ran at 1b10c9a. `inl_fast_v1` started
  while the HAQ_FAST edit was uncommitted. The NEAR and FAST code paths are unchanged in that edit,
  which was committed later as bc6869b. Round 2 ran at 572dbfd.
