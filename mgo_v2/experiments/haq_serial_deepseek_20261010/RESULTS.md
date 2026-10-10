# Stage 2 result (DeepSeek): decode miss-placement policies under serial GEMM, no H2D overlap

On DeepSeek-V2-Lite, hit-aware quota (HAQ) lowers decode TPOT by **4.32% (C50)**
and **2.64% (C30)** versus BW. These are medians of 4 unfiltered targets. Both
cache sizes show the order BW ≈ BR ≈ STATIC_BLOCK > LA_CA_NEAR > HAQ. It is the
same mechanism as in the Qwen3 result (`../haq_serial_decode_20261010`): HAQ
flattens the rank-0 copy gradient left by the balanced quota.

## Setting

- **Model and workload:** DeepSeek-V2-Lite-Chat (26 MoE layers, 64 routed
  experts, top-6, 2 shared experts on GPU). R4 on physical GPUs 0/1/4/5,
  ShareGPT, rank-local B16, input 512, output 64.
- **Cells:** the existing frozen `deepseek_cache_ablation_20261009/WORKLOADS.json`
  cells at C50 and C30. No sample seed was selected.
- **Runtime:** native serial per-expert executor. The serial H2D ablation makes
  the host wait for all of a layer's demand copies before expert compute.
  Prefetch is off. Prefill placement is BR, unlike the earlier DeepSeek headline
  runs, which used Near prefill.
- **Worker:** `examples/headline_ours_deepseek_multipolicy_worker.py` loads the
  model once per cell. Every target starts from an empty expert cache, and the
  decode policy switches at decode step 1. Target k uses the k-th entry of the
  schedule.
  - `haq_serial`: BW, BR, HAQ, LA_CA_NEAR, LA_CA_NEAR, HAQ, BR, BW
  - `static`: BW, STATIC_BLOCK, HAQ, HAQ, STATIC_BLOCK, BW (run later)
- **STATIC_BLOCK** (policy code 18) places every miss on rank
  `expert_id // (E/R)`. It does not keep the ±1 miss quota.
- No sample was discarded.

## Results (TPOT, s/token)

### DeepSeekV2Lite_ShareGPT_R4_C50_B16_L512_O64

| Policy | Targets (job: order → TPOT) | n | Median | vs BW | Decode copies by rank (first target) |
|---|---|---|---|---|---|
| BW | haq_serial_c50:1→0.3187, haq_serial_c50:8→0.3158, static_c50:1→0.3181, static_c50:6→0.3183 | 4 | 0.3182 | +0.00% (+0.0 ms) | [14271, 13860, 13458, 13018] |
| BR | haq_serial_c50:2→0.3201, haq_serial_c50:7→0.3151 | 2 | 0.3176 | -0.19% (-0.6 ms) | [14245, 13825, 13413, 13010] |
| STATIC_BLOCK | static_c50:2→0.3202, static_c50:5→0.3164 | 2 | 0.3183 | +0.02% (+0.1 ms) | [13711, 13907, 12896, 14237] |
| LA_CA_NEAR | haq_serial_c50:4→0.3100, haq_serial_c50:5→0.3089 | 2 | 0.3094 | -2.76% (-8.8 ms) | [14247, 13809, 13420, 13006] |
| HAQ | haq_serial_c50:3→0.3035, haq_serial_c50:6→0.3030, static_c50:3→0.3054, static_c50:4→0.3071 | 4 | 0.3045 | -4.32% (-13.8 ms) | [13695, 13642, 13601, 13537] |

### DeepSeekV2Lite_ShareGPT_R4_C30_B16_L512_O64

| Policy | Targets (job: order → TPOT) | n | Median | vs BW | Decode copies by rank (first target) |
|---|---|---|---|---|---|
| BW | haq_serial_c30:1→0.3491, haq_serial_c30:8→0.3433, static_c30:1→0.3431, static_c30:6→0.3454 | 4 | 0.3444 | +0.00% (+0.0 ms) | [19637, 19229, 18836, 18417] |
| BR | haq_serial_c30:2→0.3454, haq_serial_c30:7→0.3461 | 2 | 0.3458 | +0.41% (+1.4 ms) | [19603, 19181, 18772, 18396] |
| STATIC_BLOCK | static_c30:2→0.3431, static_c30:5→0.3413 | 2 | 0.3422 | -0.63% (-2.2 ms) | [19308, 19249, 18772, 18924] |
| LA_CA_NEAR | haq_serial_c30:4→0.3404, haq_serial_c30:5→0.3413 | 2 | 0.3409 | -1.01% (-3.5 ms) | [19606, 19198, 18804, 18430] |
| HAQ | haq_serial_c30:3→0.3349, haq_serial_c30:6→0.3351, static_c30:3→0.3378, static_c30:4→0.3354 | 4 | 0.3353 | -2.64% (-9.1 ms) | [19106, 19040, 18973, 18887] |

Each target is written as `job:order→TPOT`. "Decode copies by rank" is rank
0→3 over the 63 decode steps of the first target of each policy.

## Notes

- **Reproducibility:** BW and HAQ ran in both jobs of a cell. Their tokens
  matched across jobs (identical `argmax_hash` per policy), and their TPOT
  stayed within 1%.
- **Absolute gain per token is close to Qwen3's.** A remainder copy plus its
  expert costs about 0.35 + 0.2 ms × 26 layers ≈ 14 ms per token for DeepSeek,
  against 0.18 + 0.2 ms × 48 layers ≈ 18 ms for Qwen3. The percentage is larger
  because DeepSeek's TPOT is smaller.
- **Earlier DeepSeek runs showed no policy effect** because they used the
  overlapped default runtime, where demand H2D is hidden. There, TPOT barely
  changed from C20 to C50.

## Caveats

- Each policy has only 2–4 targets.
- Tokens differ between policies, so this is an end-to-end comparison on the
  same prompts, not a fixed-route comparison.
- This is the serial, non-overlapped ablation mode.
- The `source_commit` recorded in `raw/*/status.json` is the worktree HEAD at
  launch. The worker and STATIC_BLOCK code were uncommitted at that time and are
  committed here unchanged.

## Reproduce

`run_ds.sh` and `run_static.sh` (DeepSeek part) hold the exact commands; the
multi-policy options are passed through environment variables.
`python summarize.py` rebuilds `SUMMARY.json` and the tables above.
