# DeepSeek-V2-Lite R8 serial GEMM + inline H2D: quota and placement baselines

Workload `DeepSeekV2Lite_ShareGPT_R8_C30_B16_L512_O64` (`build_workload.py`). It uses the same 128 ShareGPT
conversations as the Qwen R8 cell, DeepSeek tokenization, cache 30% (499 slots, 62-63 per GPU).
Mode, identical for every arm: serial per-expert native executor; host waits for all of a layer's
demand H2D before expert compute (H2D still overlaps the forward all-to-all); inline one-pass
copy submission; prefetch OFF. Prefill policy is BR for every arm, and the decode policy switches at
decode step 1. All targets use the frozen NEAR teacher continuation (`FROZEN_TOKENS.json`, from
`dsteacher_full_v1`); the actual argmax is still recorded.

| Arm | Stage 1 quota | Stage 2 expert placement |
|---|---|---|
| NEAR (`LA_CA_NEAR`) | balanced, remainder to ranks 0,1,.. (slow GPUs 0-3) | Near |
| FAST (`NEAR_FAST`) | balanced, remainder to fast GPUs 4-7 | Near |
| FAST+Random (`FAST_RANDOM`) | FAST | uniform random within quota |
| RANDOM_QUOTA (`RANDOM_QUOTA_NEAR`) | unbalanced, i.i.d. uniform rank per miss | Near |

Three rounds with rotated order, 2 target repeats per job (6 samples per arm). NEAR is the baseline.
