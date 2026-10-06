# Completed R4 strict prefill + decode32 results

Owner stopped further R8 validation. R4 final validation is complete: four conditions, three policies,33 primary runs (132 rank receipts). All primary repeats are retained. S1 had32 paired selection candidates (64 one-shot policy timings); those are stored separately and excluded from final estimates. Static remains on hold; no diagnostic passes were run.

Units: TTFT/E2E seconds; TPOT seconds per decode step. Metrics are independently maximized over four ranks, so aggregate TTFT +32*TPOT need not equal aggregate E2E exactly.32 decode steps follow the prefill output. Same policy across both phases; cache carryover retained. GPUs0/1/4/5, C30, BF16.

## Every primary repeat

### B16 / input256

| Policy | Repeat | TTFT s | TPOT s/token | E2E s |
|---|---:|---:|---:|---:|
| BR | 1 | 5.938303 | 1.734270 | 61.434745 |
| BR | 2 | 5.800681 | 1.723653 | 60.957535 |
| BR | 3 | 5.955350 | 1.736259 | 61.515458 |
| LA_CA_NEAR | 1 | 5.737154 | 1.719139 | 60.749490 |
| LA_CA_NEAR | 2 | 5.821696 | 1.706808 | 60.439241 |
| LA_CA_NEAR | 3 | 5.787077 | 1.722172 | 60.896567 |
| LA | 1 | 5.878343 | 1.727416 | 61.155556 |
| LA | 2 | 5.740133 | 1.718039 | 60.717199 |
| LA | 3 | 5.906126 | 1.721889 | 61.006560 |

### B16 / input512

| Policy | Repeat | TTFT s | TPOT s/token | E2E s |
|---|---:|---:|---:|---:|
| BR | 1 | 8.126585 | 1.698600 | 62.481789 |
| BR | 2 | 8.197803 | 1.702001 | 62.661839 |
| LA_CA_NEAR | 1 | 7.956913 | 1.679708 | 61.707557 |
| LA_CA_NEAR | 2 | 8.033510 | 1.688411 | 62.062670 |
| LA | 1 | 8.036278 | 1.683260 | 61.900608 |
| LA | 2 | 8.073289 | 1.686956 | 62.055896 |

### B64 / input256

| Policy | Repeat | TTFT s | TPOT s/token | E2E s |
|---|---:|---:|---:|---:|
| BR | 1 | 14.196075 | 2.352965 | 89.490028 |
| BR | 2 | 14.014163 | 2.342918 | 88.986412 |
| BR | 3 | 14.024858 | 2.341355 | 88.948106 |
| LA_CA_NEAR | 1 | 13.898165 | 2.324208 | 88.271915 |
| LA_CA_NEAR | 2 | 14.017506 | 2.321479 | 88.303619 |
| LA_CA_NEAR | 3 | 13.584838 | 2.328575 | 88.093945 |
| LA | 1 | 14.204579 | 2.332975 | 88.858719 |
| LA | 2 | 13.696616 | 2.327915 | 88.188983 |
| LA | 3 | 14.251211 | 2.330857 | 88.837615 |

### B64 / input512

| Policy | Repeat | TTFT s | TPOT s/token | E2E s |
|---|---:|---:|---:|---:|
| BR | 1 | 26.164574 | 2.368085 | 101.942973 |
| BR | 2 | 26.084876 | 2.374689 | 102.074670 |
| BR | 3 | 26.171412 | 2.378433 | 102.280939 |
| LA_CA_NEAR | 1 | 25.525383 | 2.353399 | 100.833121 |
| LA_CA_NEAR | 2 | 24.888229 | 2.357812 | 100.337236 |
| LA_CA_NEAR | 3 | 25.112812 | 2.365101 | 100.795160 |
| LA | 1 | 25.261933 | 2.355181 | 100.626771 |
| LA | 2 | 24.979163 | 2.356239 | 100.377921 |
| LA | 3 | 25.453267 | 2.346405 | 100.537234 |

## Final gains relative to BR

Two repeats use their mean; three use their median independently per metric. No latency-based exclusions.

| B | Input | Candidate | TTFT gain | TPOT gain | E2E gain |
|---:|---:|---|---:|---:|---:|
| 16 | 256 | LA_CA_NEAR | +2.547% | +0.873% | +1.115% |
| 16 | 256 | LA | +1.010% | +0.714% | +0.697% |
| 16 | 512 | LA_CA_NEAR | +2.046% | +0.955% | +1.097% |
| 16 | 512 | LA | +1.316% | +0.894% | +0.949% |
| 64 | 256 | LA_CA_NEAR | +0.903% | +0.799% | +0.803% |
| 64 | 256 | LA | -1.281% | +0.515% | +0.167% |
| 64 | 512 | LA_CA_NEAR | +4.020% | +0.711% | +1.254% |
| 64 | 512 | LA | +3.450% | +0.821% | +1.506% |

Third-repeat reasons: B16/L256 BR TTFT2.345% and LA TTFT2.379%; B64/L256 LA TTFT3.641%; B64/L512 LA_CA_NEAR TTFT2.528%. All policies in each affected cell received the same third repeat. B16/L512 stopped after two. None met the >5% instability criterion, but sub-percent gains can be comparable to repeat variability and are not statistical confidence claims.

All CPU copy/state checks,33-token parity, both1584-barrier counters per rank and no compilation during primary measurements passed. H2D durations were not measured separately, so rank-specific contention cannot be determined.

This is deliberately BR-adversarial, prefill-selected best-seed headroom, not average-case performance. TPOT includes policy-specific prefill cache-state carryover. R4 LA_CA_NEAR TTFT gains span0.90–4.02%; TPOT0.71–0.96%; E2E0.80–1.25%. LA has the smaller TPOT/E2E in B64/L512.

Raw precision, full repeat ranges (derivable from all samples), all132 rank measurements, warm receipts, S1 selection pairs and source hashes are preserved in R4_COMPLETE_RECORDS.json. CSV files provide the33 primary and132 rank rows.
