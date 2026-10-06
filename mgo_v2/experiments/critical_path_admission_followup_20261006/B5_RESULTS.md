# B5 post-expert global barrier results

Status: COMPLETE. C30, local B128, R4 GPUs 0/1/4/5, frozen decode64, BF16 H1b V3/P2/T2.

Canonical token, controller, H2D copy/byte and payload parity passed. Every full64 rank performed exactly 3072 post-expert barriers. No copy-parity relaxation or automatic retry.

| Policy | TPOT (s) | E2E (s) | Repeats | TPOT range (s) |
|---|---:|---:|---:|---|
| BR | 1.481771 | 105.277789 | 2 | [1.47828271484375, 1.485260009765625] |
| FCA | 1.542861 | 110.786828 | 3 | [1.5368450927734374, 1.581003173828125] |

FCA TPOT gain versus BR: -4.123% (negative means slower).

## Separate first8 diagnostic

| Policy | Local completion (ms/step) | Global barrier (ms/step) | Return host range (ms/step) | Return NCCL residency (ms/step) | Idle barrier reference ×48 (ms/step) |
|---|---:|---:|---:|---:|---:|
| BR | 3.330 | 148.084 | 75.765 | 28.863 | 3.398 |
| FCA | 3.209 | 150.756 | 66.405 | 31.561 | 3.004 |

These are mean rank-local instrumented durations, not a sum of global critical-path penalties. Return host duration and actual return NCCL residency are different quantities; both are preserved in B5_DIAGNOSTIC_RESULTS.json. The idle barrier reference includes scheduling and is not an exact service-cost subtraction. First8 diagnostic/full64 TPOT normalization is descriptive.

The barrier is an isolation tool, not an accepted production optimization. Historical no-barrier timings are not a contemporaneous causal control. No C60, R8, H2 or Stage C extension was run.
