# B3 host executor repair

Status: REPAIR_GATE_FAIL

Clean TPOT point-estimate reduction (H1b vs H0): BR 19.57%, FCA 23.09%; timing stability passed: True.
The declared full-loop repair threshold remains binding independently of any physical gain. C60 is gated on all repair criteria.

R4 GPUs 0/1/4/5, C30/B128, frozen64, BF16 V3 P2/T2. Other users’ GPUs untouched.
Diagnostics use the fixed first8 decode-step prefix; clean primary E2E/TPOT uses all64 decode steps. Diagnostic means are not full64 timing estimates.

H0 is unchanged; H1b replays the exact expert kernel against live slot weights.
Full64 discovery/capture/validation occurs before any timing; no new graph entry or compile is permitted during measurement.

| Policy | Host call reduction | Expert host-loop reduction | Counter/packet parity |
|---|---:|---:|---|
| BR | 85.57% | 36.54% | True |
| FCA | 85.68% | 37.47% | True |

Original H1 also failed the 40% full-loop gate:
- BR: launch reduction 72.94%, host-loop reduction 34.92%.
- FCA: launch reduction 73.67%, host-loop reduction 33.59%.

## Remaining host path

| Policy/executor | Loop | Kernel launch | Gather | Weight multiply | Ready selection incl. wait | Host ready wait | Slot-use record |
|---|---:|---:|---:|---:|---:|---:|---:|
| BR/H0 | 994.69 | 616.43 | 111.75 | 111.39 | 26.22 | 0.18 | 61.67 |
| BR/H1b | 631.21 | 88.94 | 79.55 | 161.29 | 161.55 | 127.73 | 69.00 |
| FCA/H0 | 997.89 | 613.27 | 112.34 | 113.61 | 28.41 | 1.66 | 62.28 |
| FCA/H1b | 623.96 | 87.84 | 78.88 | 158.54 | 160.95 | 128.02 | 67.65 |

Values are diagnostic mean-rank ms/decode step. Ready wait is nested inside ready selection: do not add both. Other host-loop time includes diagnostic bookkeeping. This does not isolate a unique GIL/driver/OS cause.

Removing expert invocation overhead exposes more readiness waiting. Gather/weighting and slot-use recording remain material; CUDA-graph replay alone does not establish the required full-loop repair. No grouped GEMM or scheduler change is authorized here.

| Policy/executor | TPOT (s) | E2E (s) |
|---|---:|---:|
| BR_H0 | 1.825885 | 127.089899 |
| FCA_H0 | 1.979427 | 138.621437 |
| BR_H1b | 1.468470 | 104.297345 |
| FCA_H1b | 1.522476 | 109.411342 |

Observed TPOT reduction: BR 19.57%; FCA 23.09%. Stability gate: True. These are physical timing observations; the full repair gate still requires every host/correctness/stability criterion.

All valid repeats and full ranges are retained in the accompanying JSON/CSV. Two stable repeats stop; at most one conditional third. No noise-based deletion.

C60 authorization: False. Stage C oracle remains unauthorized.

Nsight node tracing is separate from timing. H1 launch ranges include scratch copies and replay; host and GPU spans overlap and must not be added.
Runtime repair is not an admission-method contribution. See graph signatures, correctness receipts and diagnostic source hashes.

Clean comparison graph cache: 38289–42135 exact signatures/rank; max persistent scratch 36.18 GiB/rank. H0 retains the same allocated graph buffers but does not replay them.
Graph discovery/capture is benchmark preparation, outside E2E/TPOT. Unseen signatures invalidate the run; this is not a deployable dynamic-shape executor claim.

Resource audit: peak worker GPU allocation 51.09 GiB; minimum recorded host available memory 1617.01 GiB. Primary boundary snapshots show no foreign GPU jobs. Only physical GPUs 0/1/4/5 were used; owned model inference workers were restored on those GPUs.
