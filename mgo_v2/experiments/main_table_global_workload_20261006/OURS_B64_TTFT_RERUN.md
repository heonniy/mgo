# Owner-requested OURS B64/L512 TTFT rerun

Owner explicitly requested one additional TTFT measurement after reviewing the main-table results.

- R4, physical GPUs 0/1/4/5 only, C30, local batch64/global256, input512.
- Unchanged Near/H0/full-pinned runtime and frozen target/warmup manifests.
- One disjoint full64-token warmup, then reset dynamic expert state and run exactly one full64-token primary. TTFT is the requested outcome; other native timing receipts remain preserved.
- Added optional repeats argument to the existing supervisor/OURS worker; the default remains three. No runtime or measurement-clock changes.
- Raw directory: `/home/hwlee/mgo-results/headline_r4_20261007/ours_B64_L512_owner_ttft_rerun1`.
- This separate single-shot follow-up does not replace any previous sample or the main-table stability audit.

Status: PASS. All four rank receipts pass cache consistency, cold expert state, finite logits and no-compilation checks.

| Measurement | TTFT seconds | TPOT seconds/token | E2E seconds |
|---|---:|---:|---:|
| Disjoint warmup | 27.470038 | 1.161000 | 100.613059 |
| Single primary | 24.240570 | 1.013278 | 88.077098 |

This single sample does not establish timing stability or resolve the prior scaling/variability cause. Original results remain unchanged. Owned model loads restored on GPUs0/1/4/5 after worker exit.

