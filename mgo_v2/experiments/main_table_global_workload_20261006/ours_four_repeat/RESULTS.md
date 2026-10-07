# Four-repeat OURS TTFT / TPOT / E2E

Owner-requested exactly4 clean repeats per cell. R4/C30 on GPUs0/1/4/5, Near/H0/full-pinned, prefill-optimized and prefill-layout-fast, output64. Same frozen ShareGPT-long inputs; local B16/global64 with input256 and local B64/global256 with input512. No runtime changes beyond allowing a4-repeat count. No extra fifth repeat or outlier deletion.

Preparation: one-token index/metadata validation on disjoint warmup inputs, full64-token disjoint warmup, then4 measured batches with expert/cache/history reset before each. Compiled kernels and CPU pinned backing retained. No post-primary diagnosis/profiler. All common-release/max-rank timing rules unchanged. TPOT=(E2E−TTFT)/63.

## All measured samples

Seconds; first repeat is retained.

| Cell | Repeat | TTFT | TPOT | E2E |
|---|---:|---:|---:|---:|
| B16_L256 | 1 | 3.311342 | 0.753095 | 50.756335 |
| B16_L256 | 2 | 1.587979 | 0.760180 | 49.479294 |
| B16_L256 | 3 | 1.586388 | 0.748997 | 48.773188 |
| B16_L256 | 4 | 1.602408 | 0.747791 | 48.713254 |
| B64_L512 | 1 | 5.964081 | 0.972013 | 67.200900 |
| B64_L512 | 2 | 4.401916 | 0.959923 | 64.877057 |
| B64_L512 | 3 | 4.419386 | 0.964165 | 65.161798 |
| B64_L512 | 4 | 4.415372 | 0.969910 | 65.519718 |

## Mean ± sample standard deviation

n=4, denominator n−1. All4 samples contribute independently to every metric.

| Cell | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| B16_L256 | 2.022029 ± 0.859572 | 0.752516 ± 0.005591 | 49.430518 ± 0.949863 |
| B64_L512 | 4.800189 ± 0.775964 | 0.966503 ± 0.005499 | 65.689868 ± 1.041103 |

## Range and timing spread

Spread=(max−min)/mean. A whole row is timing-stable only if all three spreads are at most5%; valid execution does not imply stable timing.

| Cell | Metric | Min | Max | Median | Spread |
|---|---|---:|---:|---:|---:|
| B16_L256 | TTFT | 1.586388 | 3.311342 | 1.595194 | 85.31% |
| B16_L256 | TPOT | 0.747791 | 0.760180 | 0.751046 | 1.65% |
| B16_L256 | E2E | 48.713254 | 50.756335 | 49.126241 | 4.13% |
| B64_L512 | TTFT | 4.401916 | 5.964081 | 4.417379 | 32.54% |
| B64_L512 | TPOT | 0.959923 | 0.972013 | 0.967038 | 1.25% |
| B64_L512 | E2E | 64.877057 | 67.200900 | 65.340758 | 3.54% |

## Verification and limits

All GPU index/metadata checks, finite logits, cache consistency and no-compilation guards pass. Every generated token is compared against the preceding layout-repaired primary on identical request IDs. The unchanged runtime has no intended policy or arithmetic change.

- B16_L256: 16384 token positions identical; final cache roles/state/controller equality=True; whole-row timing stable=False; minimum host available=1388.72GiB.
- B64_L512: 65536 token positions identical; final cache roles/state/controller equality=True; whole-row timing stable=False; minimum host available=1404.29GiB.

Repeats run within one loaded process per cell; they do not estimate variation across independent process launches or hosts. Host/model warm state beyond the reset expert cache remains. This packet does not diagnose the cause of first-repeat differences. Earlier results are preserved and not pooled into these statistics. Source commits and archive checksums are in RESULTS.json; full resource logs remain in the raw directories. Owned model loads restored on0/1/4/5 after completion.
