# Original BF16 prefix screen completed

Both batches finished all nine BR settings. BF16 only. No numerical-reference runs. All raw repetitions remain archived. Legacy T2 is ineligible because its host trigger did not wait for physical forward completion; see T2_COMPLETION_REPAIR.md.

| Batch | Setting | Repeats | Timing gate | E2E estimate (s) | TPOT estimate (s) |
|---|---|---|---|---|---|
| 128 | P1_T1 | 7 | UNRESOLVED_JITTER | 92.248315 | 1.228133 |
| 128 | P1_T0 | 2 | STABLE | 89.063518 | 1.194355 |
| 128 | P1_T2 | 2 | STABLE | 89.108665 | 1.188175 |
| 128 | P2_T2 | 7 | UNRESOLVED_JITTER | 91.406899 | 1.227130 |
| 128 | P2_T1 | 7 | UNRESOLVED_JITTER | 91.024862 | 1.226332 |
| 128 | P2_T0 | 7 | UNRESOLVED_JITTER | 87.011320 | 1.165880 |
| 128 | P4_T0 | 2 | STABLE | 85.958784 | 1.145620 |
| 128 | P4_T2 | 2 | STABLE | 94.020070 | 1.273091 |
| 128 | P4_T1 | 7 | UNRESOLVED_JITTER | 90.190099 | 1.212143 |
| 256 | P4_T1 | 3 | UNRESOLVED_JITTER_FUTILITY | 120.287834 | 1.511854 |
| 256 | P4_T2 | 3 | UNRESOLVED_JITTER_FUTILITY | 115.345153 | 1.429444 |
| 256 | P4_T0 | 4 | UNRESOLVED_JITTER_FUTILITY | 117.341641 | 1.461136 |
| 256 | P2_T0 | 2 | STABLE | 113.749280 | 1.409864 |
| 256 | P2_T1 | 5 | UNRESOLVED_JITTER_FUTILITY | 116.788982 | 1.440561 |
| 256 | P2_T2 | 6 | UNRESOLVED_JITTER_FUTILITY | 114.717797 | 1.420861 |
| 256 | P1_T2 | 2 | STABLE | 115.867140 | 1.441883 |
| 256 | P1_T0 | 2 | STABLE | 115.422544 | 1.435302 |
| 256 | P1_T1 | 3 | STABLE | 113.485533 | 1.402019 |

P1/T0 is stable in both batches and unaffected by the T2 repair. Final BR-only selection awaits the corrected T2 screen and then full-256 confirmation. This is not a BR-vs-LA gain result.
