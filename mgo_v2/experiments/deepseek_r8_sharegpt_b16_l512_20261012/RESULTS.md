# DeepSeek-V2-Lite R8 serial GEMM + inline H2D: quota and placement baselines

Setup: see PLAN.md (serial per-expert, no H2D/compute overlap, inline one-pass H2D, prefetch OFF,
frozen NEAR teacher tokens, BR prefill for all arms). Three rounds with rotated order,
6 samples per arm. Full table: `RESULTS_TABLE.md` (`python summarize_ds.py`).

| Arm | Stage 1 quota | Stage 2 placement | TPOT median [range], s | vs NEAR | per round |
|---|---|---|---:|---:|---|
| NEAR | balanced, remainder to GPUs 0-3 | Near | 0.2576 [0.2551, 0.2589] | 0 | — |
| FAST | balanced, remainder to GPUs 4-7 | Near | 0.2520 [0.2493, 0.2548] | **-2.16%** | -1.45, -2.49, -2.13 |
| FAST+Random | FAST | random | 0.2552 [0.2530, 0.2572] | -0.91% | -0.71, -0.77, -0.88 |
| RANDOM_QUOTA | unbalanced random | Near | 0.2958 [0.2919, 0.2984] | **+14.83%** | +14.20, +14.51, +15.68 |

Paired by round and repeat position (all six pairs agree in every row):
- FAST is faster than NEAR in 6/6 pairs.
- FAST+Near is faster than FAST+Random in 6/6 pairs; the median pair gap is about 1.3%.
- RANDOM_QUOTA is slower than NEAR in 6/6 pairs.

Mechanism and comparison with Qwen3 R8 (same mode, same conversations):

| | Qwen3-30B-A3B | DeepSeek-V2-Lite |
|---|---:|---:|
| FAST vs NEAR (stage 1, +1 placement) | -0.5% (not measurable) | **-2.2%** |
| Near vs random placement (stage 2) | -2.2% | -1.3% |
| random unbalanced quota vs NEAR | +11.4% | +14.8% |

- DeepSeek experts are 16.5 MiB versus 9 MiB for Qwen, so H2D is a larger share of each serial
  layer. Moving the remainder copies to the fast GPUs then shows up in TPOT.
  The max H2D wait goes from 3.22 s (GPUs 0-3) under NEAR to 2.91 s under FAST.
- Balance is the dominant stage-1 property in both models. RANDOM_QUOTA's whole-run per-rank fetches
  are the most even (9.9-10.1k), yet each layer has a busiest rank, and every rank waits for it.
- Teacher-token divergence (actual argmax vs teacher) is 137-147 of 8,192 positions for non-NEAR arms.
- Provenance: the source tree had another session's uncommitted, default-preserving NEAR_BPS
  edit in `scripts/env_offload_policy.py` (`source_dirty=true` in every job status). With
  MGO_NEAR_BPS unset it is behaviour-identical (200 bps).
