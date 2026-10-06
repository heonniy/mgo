# Owner-stopped ShareGPT_LONG512 TTFT analysis

Status: OWNER_STOPPED_ANALYZED. The owner stopped the final C60/B128 condition and requested analysis of available results. S2 and phase diagnostics were not run. The generic runner labels the stop FAIL with `owner STOP`; this is an intentional cancellation, not a detected correctness/OOM failure. Raw receipts remain unchanged.

## Scope and coverage

R4 physical GPUs 0/1/4/5; BF16; exact512 valid tokens from 2048 frozen long ShareGPT conversation prefixes; empty expert cache; substitution/replication off. CA and OLD_CA are separate candidates, with paired BR and LA.

| Cache | Local batch | Complete pairs / planned | Status |
|---|---:|---:|---|
| C30 | 16 | 24/24 | COMPLETE_S1 |
| C60 | 16 | 24/24 | COMPLETE_S1 |
| C30 | 128 | 24/24 | COMPLETE_S1 |
| C60 | 128 | 13/24 | OWNER_STOPPED_PARTIAL |

## Best observed selection candidates

These are single-shot S1 best-seed observations, **not validated primary gains**. Each candidate has its own selected workload/placement seed and its own matched BR time. Do not compare absolute times between different selected workloads as a causal policy comparison.

| Cache | Batch | Policy | Candidates completed | Seed (workload, placement) | BR TTFT s | Policy TTFT s | Best gain % | All observed gain range % |
|---|---:|---|---:|---|---:|---:|---:|---|
| C30 | 16 | CA | 8/8 | (24, 20) | 7.759 | 8.397 | -8.22 | -9.78 to -8.22 |
| C30 | 16 | OLD_CA | 8/8 | (24, 20) | 7.805 | 8.448 | -8.24 | -9.47 to -8.24 |
| C30 | 16 | LA | 8/8 | (19, 9) | 7.921 | 7.726 | +2.45 | +1.02 to +2.45 |
| C60 | 16 | CA | 8/8 | (24, 27) | 7.827 | 8.506 | -8.68 | -9.93 to -8.68 |
| C60 | 16 | OLD_CA | 8/8 | (24, 20) | 7.818 | 8.519 | -8.96 | -9.78 to -8.96 |
| C60 | 16 | LA | 8/8 | (15, 12) | 7.881 | 7.705 | +2.23 | +1.44 to +2.23 |
| C30 | 128 | CA | 8/8 | (27, 27) | 47.488 | 50.070 | -5.44 | -8.70 to -5.44 |
| C30 | 128 | OLD_CA | 8/8 | (9, 27) | 47.828 | 48.137 | -0.65 | -9.52 to -0.65 |
| C30 | 128 | LA | 8/8 | (0, 28) | 47.371 | 46.148 | +2.58 | +1.34 to +2.58 |
| C60 | 128 | CA | 5/8 | (27, 24) | 48.003 | 49.074 | -2.23 | -4.92 to -2.23 |
| C60 | 128 | OLD_CA | 4/8 | (27, 6) | 47.863 | 49.316 | -3.03 | -4.99 to -3.03 |
| C60 | 128 | LA | 4/8 | (6, 28) | 48.089 | 47.142 | +1.97 | +1.56 to +1.97 |

## Work accounting and interpretation

All completed pairs pass CPU physical-copy/state checks, cross-policy first-token parity, warm-to-measure token parity and no compilation during measurement. Each pair has equal per-rank expert H2D bytes/copies and equal total routed expert rows. No incomplete pair is used.

For the same C30/B16 workload24 / placement20 pair:
- BR -> CA TTFT: 7.759031 -> 8.397091 s.
- Payload wire bytes: 52.503 -> 43.819 GiB (-16.54%). This counter excludes metadata collective traffic.
- Sum over layers of maximum-rank expert rows: 3,679,777 -> 5,853,205 (+59.06%).
- Rank expert rows: BR [3146526, 3132716, 3216289, 3087381]; CA [5853205, 2614416, 1985672, 2129619].
- Total H2D identical at 53.921 GiB.

CA constrains the number of admitted experts per rank while optimizing locality; that does not balance token work per expert. The measured routing concentrates hot-expert work on a rank. Lower payload traffic together with higher maximum-rank work is consistent with compute imbalance offsetting communication savings. Old CA has a similar cold-prefill assignment pattern. LA balances token work and shows modest observed gains. This is a mechanism hypothesis supported by work counts, not a measured decomposition of the slowdown; controller cost, GEMM shapes and synchronization can also contribute.

Cache30/60 both start empty and visit each layer once: current-layer experts require the same mandatory transfers. Larger capacity changes retained/victim state, not a current-layer cache-hit benefit. Do not generalize this result to repeated decode or warm-cache inference.

## Decision

No stable GO/STOP threshold is claimed because S2 was canceled. Among the observed favorable seed screens, LA is the most promising candidate; CA/OLD_CA did not demonstrate positive headroom in the completed observations. No production-policy change is justified by this selection-only dataset. C60/B128 remains explicitly partial. No new GPU experiments are queued. Owned resident model loads are restored only on0/1/4/5;2/3/6/7 remain untouched.

Full pair-level metrics, audited raw receipt paths/hashes, all observed gain ranges and interrupted-pair inventory are in `TTFT_STOPPED_RESULTS.json`.
