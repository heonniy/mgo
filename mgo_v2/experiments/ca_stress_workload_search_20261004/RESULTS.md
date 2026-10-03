# CA communication-stress search

These are optimized real-request stress examples, not dataset-average results.
Exact frozen-route resource accounting only: no physical E2E or quality claim.
Primary sample/DP choices maximize median absolute peer savings across eight
predeclared BR seeds. Best observed BR seed is reported separately.

| Dataset | R | Local B | Cache | Sample / DP | Median peer reduction | Best observed BR seed |
|---|---:|---:|---:|---|---:|---:|
| MATH | 4 | 8 | 30% | 129 / 125 | 22.74% | 131 |
| MATH | 4 | 8 | 40% | 129 / 125 | 20.07% | 131 |
| MATH | 4 | 8 | 50% | 129 / 125 | 17.71% | 73 |
| MATH | 4 | 8 | 60% | 129 / 125 | 16.06% | 131 |
| MATH | 4 | 16 | 30% | 129 / 208 | 18.29% | 19 |
| MATH | 4 | 16 | 40% | 129 / 208 | 16.13% | 42 |
| MATH | 4 | 16 | 50% | 129 / 208 | 14.51% | 73 |
| MATH | 4 | 16 | 60% | 129 / 208 | 13.30% | 19 |
| MATH | 4 | 32 | 30% | 20 / 2 | 13.29% | 99 |
| MATH | 4 | 32 | 40% | 20 / 2 | 11.69% | 99 |
| MATH | 4 | 32 | 50% | 20 / 2 | 10.37% | 99 |
| MATH | 4 | 32 | 60% | 20 / 2 | 9.34% | 99 |
| MATH | 4 | 64 | 30% | 37 / 153 | 9.86% | 73 |
| MATH | 4 | 64 | 40% | 37 / 153 | 8.86% | 73 |
| MATH | 4 | 64 | 50% | 37 / 153 | 8.07% | 7 |
| MATH | 4 | 64 | 60% | 37 / 153 | 7.69% | 251 |
| MATH | 8 | 8 | 30% | 168 / 168 | 16.40% | 251 |
| MATH | 8 | 8 | 40% | 168 / 168 | 14.56% | 251 |
| MATH | 8 | 8 | 50% | 168 / 168 | 12.82% | 99 |
| MATH | 8 | 8 | 60% | 168 / 168 | 11.71% | 19 |
| MATH | 8 | 16 | 30% | 163 / 62 | 12.52% | 251 |
| MATH | 8 | 16 | 40% | 163 / 38 | 11.03% | 131 |
| MATH | 8 | 16 | 50% | 163 / 38 | 10.17% | 251 |
| MATH | 8 | 16 | 60% | 163 / 38 | 9.40% | 131 |
| MATH | 8 | 32 | 30% | 253 / 86 | 10.07% | 7 |
| MATH | 8 | 32 | 40% | 253 / 86 | 9.10% | 7 |
| MATH | 8 | 32 | 50% | 253 / 86 | 8.22% | 7 |
| MATH | 8 | 32 | 60% | 253 / 86 | 7.70% | 19 |
| MATH | 8 | 64 | 30% | 217 / 14 | 8.20% | 251 |
| MATH | 8 | 64 | 40% | 217 / 14 | 7.51% | 42 |
| MATH | 8 | 64 | 50% | 217 / 14 | 7.03% | 42 |
| MATH | 8 | 64 | 60% | 8 / 228 | 6.77% | 73 |
| ShareGPT | 4 | 8 | 30% | 205 / 4 | 29.86% | 73 |
| ShareGPT | 4 | 8 | 40% | 205 / 4 | 27.16% | 19 |
| ShareGPT | 4 | 8 | 50% | 205 / 4 | 25.05% | 73 |
| ShareGPT | 4 | 8 | 60% | 205 / 4 | 23.70% | 73 |
| ShareGPT | 4 | 16 | 30% | 248 / 138 | 22.97% | 131 |
| ShareGPT | 4 | 16 | 40% | 248 / 138 | 21.27% | 131 |
| ShareGPT | 4 | 16 | 50% | 248 / 138 | 20.00% | 99 |
| ShareGPT | 4 | 16 | 60% | 248 / 138 | 18.94% | 19 |
| ShareGPT | 4 | 32 | 30% | 170 / 76 | 17.53% | 19 |
| ShareGPT | 4 | 32 | 40% | 170 / 76 | 16.33% | 19 |
| ShareGPT | 4 | 32 | 50% | 170 / 76 | 15.59% | 19 |
| ShareGPT | 4 | 32 | 60% | 170 / 76 | 14.86% | 19 |
| ShareGPT | 4 | 64 | 30% | 242 / 202 | 12.48% | 73 |
| ShareGPT | 4 | 64 | 40% | 242 / 202 | 11.60% | 131 |
| ShareGPT | 4 | 64 | 50% | 242 / 202 | 11.11% | 19 |
| ShareGPT | 4 | 64 | 60% | 242 / 202 | 11.00% | 73 |
| ShareGPT | 8 | 8 | 30% | 81 / 86 | 20.97% | 42 |
| ShareGPT | 8 | 8 | 40% | 81 / 86 | 19.31% | 19 |
| ShareGPT | 8 | 8 | 50% | 81 / 86 | 17.75% | 73 |
| ShareGPT | 8 | 8 | 60% | 81 / 86 | 16.74% | 42 |
| ShareGPT | 8 | 16 | 30% | 254 / 234 | 16.38% | 251 |
| ShareGPT | 8 | 16 | 40% | 180 / 57 | 15.41% | 7 |
| ShareGPT | 8 | 16 | 50% | 180 / 57 | 14.46% | 7 |
| ShareGPT | 8 | 16 | 60% | 180 / 57 | 13.88% | 19 |
| ShareGPT | 8 | 32 | 30% | 113 / 5 | 12.61% | 19 |
| ShareGPT | 8 | 32 | 40% | 113 / 5 | 11.86% | 7 |
| ShareGPT | 8 | 32 | 50% | 113 / 5 | 11.56% | 7 |
| ShareGPT | 8 | 32 | 60% | 113 / 5 | 11.33% | 7 |
| ShareGPT | 8 | 64 | 30% | 242 / 8 | 10.54% | 7 |
| ShareGPT | 8 | 64 | 40% | 242 / 8 | 10.00% | 7 |
| ShareGPT | 8 | 64 | 50% | 184 / 86 | 9.53% | 73 |
| ShareGPT | 8 | 64 | 60% | 184 / 30 | 9.58% | 7 |

Full H2D, hit rates, byte gains, seed ranges, quota checks and state hashes are
in primary_stress.csv. top3_candidates.csv preserves selection alternatives.
selected_manifests/ pins exact request membership and balanced rank order.
neutral_reference.csv supplies the prior unoptimized random-prefix controls;
their seed42 statistic is not an eight-seed median. Larger candidate pools and
optimized sample/rank selection intentionally favor stress cases.
No further physical timing or replica-policy experiment follows automatically.
