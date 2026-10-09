# main_OURS grouped expert A/B/C primary timing

Qwen3-30B, ShareGPT, C30, per-rank B16, input512, decode64, Near, prefetch OFF, identical frozen requests in each R. A is C++ Ready-First individual GEMMs; B waits all H2D then one grouped wave; C executes ready experts then remaining misses in a second grouped wave.

| R | Arm | TTFT s (two samples) | TPOT s/token (two samples) | E2E s (two samples) | Mean TPOT | E2E stability |
|---|---|---:|---:|---:|---:|---|
| R8 | A | 3.1413 / 1.8263 | 0.3944 / 0.3881 | 27.9866 / 26.2742 | 0.3912 | UNSTABLE_E2E |
| R8 | B | 3.1572 / 1.8488 | 0.3094 / 0.3081 | 22.6515 / 21.2586 | 0.3088 | UNSTABLE_E2E |
| R8 | C | 3.1731 / 1.8081 | 0.3081 / 0.3061 | 22.5840 / 21.0927 | 0.3071 | UNSTABLE_E2E |
| R4 | A | 3.6283 / 1.7549 | 0.5002 / 0.4996 | 35.1400 / 33.2278 | 0.4999 | UNSTABLE_E2E |
| R4 | B | 3.6495 / 1.7599 | 0.3427 / 0.3462 | 25.2386 / 23.5696 | 0.3444 | UNSTABLE_E2E |
| R4 | C | 3.6469 / 1.7563 | 0.3508 / 0.3404 | 25.7463 / 23.2043 | 0.3456 | UNSTABLE_E2E |

All samples are retained. Every arm in a given R has identical rank-level H2D bytes, copy counts, final cache-state hashes, and generated tokens in both repeats. First-target TTFT is higher in each arm; E2E exceeds the 5% repeat threshold, so E2E is marked unstable without further repetitions. TPOT remains substantially more stable.

| R | Total H2D GiB | Copies | B gain vs A | C gain vs A | C gain vs B |
|---|---:|---:|---:|---:|---:|
| R8 | 1856.681 | 211,249 | 21.08% | 21.50% | 0.54% |
| R4 | 1578.313 | 179,577 | 31.10% | 30.86% | -0.34% |

A→B changes both kernel implementation and H2D scheduling, so it does not isolate their separate effects. B→C keeps the grouped kernel and changes only readiness scheduling. The small B/C difference is within observed repeat variation, especially on R4. Instrumented breakdown is reported separately.
