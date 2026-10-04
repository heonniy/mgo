# B128 BF16 BR-only prefix screen

Nine settings completed. Four pass the declared stability rule; five remain ineligible after seven repeats. All physical correctness gates passed. This is the first 64-step prefix, not the final decode256 BR/LA comparison.

| Setting | Repeats | Status | E2E estimate (s) | TPOT estimate (s) | TPOT full range (s) |
|---|---:|---|---:|---:|---|
| P1_T1 | 7 | UNRESOLVED_JITTER | 92.248 | 1.228133 | 1.171471–1.334376 |
| P1_T0 | 2 | STABLE | 89.064 | 1.194355 | 1.192560–1.196150 |
| P1_T2 | 2 | STABLE | 89.109 | 1.188175 | 1.186608–1.189742 |
| P2_T2 | 7 | UNRESOLVED_JITTER | 91.407 | 1.227130 | 1.148892–1.363355 |
| P2_T1 | 7 | UNRESOLVED_JITTER | 91.025 | 1.226332 | 1.113389–1.320166 |
| P2_T0 | 7 | UNRESOLVED_JITTER | 87.011 | 1.165880 | 1.091203–1.223681 |
| P4_T0 | 2 | STABLE | 85.959 | 1.145620 | 1.140209–1.151032 |
| P4_T2 | 2 | STABLE | 94.020 | 1.273091 | 1.267253–1.278929 |
| P4_T1 | 7 | UNRESOLVED_JITTER | 90.190 | 1.212143 | 1.128861–1.323922 |

P4/T0 has the lowest BR TPOT among stable B128 settings. Selection remains pending B256 and full decode256 BR-only confirmation. Unstable estimates are descriptive only and cannot win. No LA gain has been measured by this screen.
