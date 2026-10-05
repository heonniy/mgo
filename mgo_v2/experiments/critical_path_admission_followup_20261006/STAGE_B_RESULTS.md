# Stage B: held-out critical-path validation

**FAIL: stop before oracle.** Offline Env1 C30/C60 × BR/OLD_CA/FCA/LA_CA, 24,576 decode events. No new GPU inference, no retiming, no exclusions, and no fitting to policy timings.

The model was committed before validation at `96f034b`. It uses nonnegative intercept + maximum rank incident packets, fitted only to A1 synthetic shapes separately for forward/return; stable A3 power-ladder tau(n) with linear row interpolation; return base + expert-ready skew. Layer score is forward + maximum expert service + return base, with no duplicated arrival wait.

## Gates

- Qualitative tau-model ordering, C30/C60: [False, True].
- All-cell critical correlation≥0.60 and phase aggregate error≤25%: {'critical_tau': False, 'critical_rows': False}.
- Tau no worse than row-only baseline in both expert correlation and aggregate error: True.

## Critical score vs measured event span

| Cache | Policy | Pred ms/step | Observed ms/step | Spearman | Aggregate error |
|---|---|---:|---:|---:|---:|
| C30 | BR | 32.070 | 1515.349 | 0.581 | 97.9% |
| C30 | OLD_CA | 36.022 | 1615.743 | 0.591 | 97.8% |
| C30 | FCA | 39.825 | 1709.643 | 0.702 | 97.7% |
| C30 | LA_CA | 31.145 | 1530.801 | 0.633 | 98.0% |
| C60 | BR | 33.042 | 1151.856 | 0.647 | 97.1% |
| C60 | OLD_CA | 35.137 | 1194.216 | 0.758 | 97.1% |
| C60 | FCA | 40.568 | 1307.625 | 0.784 | 96.9% |
| C60 | LA_CA | 32.544 | 1134.935 | 0.665 | 97.1% |

Supplemental active-phase-only comparison also fails: maximum rank-local union of forward/expert/return kernels (zero phase overlap verified for all32captures) gives aggregate errors C30/BR 90.1%, C30/OLD_CA 92.4%, C30/FCA 93.3%, C30/LA_CA 90.5%, C60/BR 90.3%, C60/OLD_CA 91.6%, C60/FCA 93.3%, C60/LA_CA 90.1%. This removes intervening CPU/H2D-only gaps and does not change the frozen primary target.

Measured target is the maximum rank-local span from that event’s first forward NCCL kernel start to last return NCCL kernel end. It is read from existing CUPTI timestamps. It includes actual intervening gaps, not a sum of expert max and return max. Summing these event spans does not define TPOT.

## Phase and baseline checks

| Cache | Policy | Expert tau corr | Row corr | Expert tau aggregate error | Return corr | H2D mean-rank ms/step |
|---|---|---:|---:|---:|---:|---:|
| C30 | BR | 0.994 | 0.112 | 23.2% | 0.676 | 168.145 |
| C30 | OLD_CA | 0.999 | 0.466 | 23.3% | 0.862 | 168.321 |
| C30 | FCA | 0.999 | 0.607 | 23.4% | 0.915 | 168.858 |
| C30 | LA_CA | 0.966 | -0.189 | 23.2% | 0.570 | 168.296 |
| C60 | BR | 0.997 | 0.135 | 24.2% | 0.760 | 84.876 |
| C60 | OLD_CA | 0.999 | 0.261 | 24.0% | 0.863 | 87.070 |
| C60 | FCA | 0.999 | 0.422 | 24.2% | 0.926 | 88.578 |
| C60 | LA_CA | 0.990 | -0.149 | 24.0% | 0.756 | 84.910 |

Return residency retains all rank/event values and p10/p50/p90 distributions in JSON. It is not pure wire time. H2D is reported separately and does not enter fitting.

Absolute forward residency aggregate errors are 98.1%–98.9%; return residency errors are 94.9%–96.5%. Matching a policy ratio does not fix this large absolute discrepancy.

## FCA return inflation
- C30: predicted FCA/BR 2.725×; observed 2.633×; consistency gate True.
- C60: predicted FCA/BR 2.374×; observed 2.417×; consistency gate True.

## H2D trajectory comparison
- C30/OLD_CA vs BR: observed H2D +0.10%; mandatory fetches +0.26%. No fitted H2D coefficient.
- C30/FCA vs BR: observed H2D +0.42%; mandatory fetches +0.15%. No fitted H2D coefficient.
- C30/LA_CA vs BR: observed H2D +0.09%; mandatory fetches +0.05%. No fitted H2D coefficient.
- C60/OLD_CA vs BR: observed H2D +2.58%; mandatory fetches +1.79%. No fitted H2D coefficient.
- C60/FCA vs BR: observed H2D +4.36%; mandatory fetches +1.33%. No fitted H2D coefficient.
- C60/LA_CA vs BR: observed H2D +0.04%; mandatory fetches +0.10%. No fitted H2D coefficient.

## Limits and interpretation

A1 establishes isolated concentration sensitivity and A2 establishes isolated skew sensitivity; those facts do not establish that service-time skew suffices in the full runtime. The model omits CPU enqueue gaps, staging/H2D waits, packet packing and routing-weight work. The measured expert attribution contains the full expert loop’s GPU operations, whereas tau measures the compiled expert function span. These are possible missing mechanisms, not newly proven root causes.

The A1 calibration covers only 1389–1528 packets; extrapolation is explicitly flagged per event. A3 uses the stable primary ladder; the unstable supplemental rank1/n18 point is not used. No coefficients were adjusted to compensate for validation error. Exact per-event split, row, fetch and final controller-state parity with the old captures/proofs must pass before a cell is included.

Detailed per-event owner/row lists remain in hashed local sidecars. Portable event split/score data and all numerical validation are committed. Ordinal predicted order matches LA_CA < BR < OLD_CA < FCA in both caches; C30 fails only the frozen 2% near-equality convention for LA_CA versus BR. The absolute-time failures independently prevent PASS regardless of that convention. Stage-B failure does not show that all admission optimization is impossible; it rejects this calibrated mechanism as justification for an exact oracle. Per STAGE_B_EXECUTION.md, stop for owner review and do not run Stage C.
