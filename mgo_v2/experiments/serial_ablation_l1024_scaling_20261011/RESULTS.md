# Serial ablation: OURS (HAQ) vs OURS-random across datasets, models, input lengths and GPU counts

OURS (HAQ, code 16) and OURS-random (RANDOM, code 19) in the serial ablation mode of
`../cache_size_sweep_ablation`. All 56 jobs passed. They ran at three commits that differ only in driver logging, drivers and the
summarizer:
- f3d69b3: `run.sh`
- 254c6bd: `run_lmsys.sh`
- a1bc375: `run_b64_random.sh`

**Main result:**
- **Every R4 cell:** RANDOM's TPOT is 11.0–12.4% higher than OURS on Qwen3 and 15.5–15.7% higher
  on DeepSeek. The gap holds across ShareGPT and LMSYS-Chat-1M and across input 512 and 1024.
- **GPU count (Qwen3 ShareGPT 512):** the gap grows with rank count: +5.8% (2 GPUs), +11.4% (4),
  +14.3% (8).
- **TTFT:** the same for both policies, as expected. Prefill is BR in both, and the policy switches
  at decode step 1.

## Setting

- **Common:** C30 expert budget, rank-local B16, output 64, greedy, no EOS stop.
- **Serial ablation mode (same flags as `../cache_size_sweep_ablation/run_*.sh`):**
  native serial per-expert executor, `--h2d-serial-ablation`, prefetch off, BR prefill,
  decode policy switched at step 1, empty expert cache at each target.
- **Part A:** R4 (GPUs 0/1/4/5); ShareGPT and LMSYS-Chat-1M × Qwen3-30B-A3B-Instruct-2507 and
  DeepSeek-V2-Lite-Chat × input 512 and 1024.
  - Manifests: the main-table `main_table_2x2_20261008/{ShareGPT,LMSYS-Chat-1M}/WORKLOADS.json`.
  - Qwen3 ShareGPT 512 used `qwen_cache_ablation_20261009` C30, whose warmup and target hashes
    equal the main-table cell's.
- **Part B:** Qwen3, ShareGPT, input 512, R2 (GPUs 0/1), R4 (0/1/4/5), R8 (0–7).
  - Manifests: `qwen_r2_sharegpt_b16_l512_20261010`, `qwen_cache_ablation_20261009` (C30),
    `qwen_r8_sharegpt_b16_l512_20261009`.
  - R4 was rerun at f3d69b3 rather than reused.
- **Part C:** the Part A matrix at local B64 (256 global requests), OURS-random only, two rounds
  per cell (`MGO_RANDOM_SALT` = round).
  - There is no serial-mode OURS (HAQ) B64 run, so Part C has no TPOT gap column.
  - The main-table B64 OURS rows use the default overlap mode and are not comparable.
- **Order (Parts A and B):** each cell runs HAQ r1, RANDOM r1, RANDOM r2, HAQ r2. `MGO_RANDOM_SALT` = round.
- **Samples:** every target is kept (repeat 0 is the in-job warmup).
  - Qwen3: 2 jobs × 3 targets = 6.
  - DeepSeek: 2 jobs × 2 targets = 4.
- **TPS** = global requests × 64 / E2E, the main-table definition.

## Part A: R4 (GPUs 0/1/4/5), C30, B16 (median [min, max], n)

| Dataset | Model | Input | GPUs | Method | TPS ↑ | TTFT (s) ↓ | TPOT (s/token) ↓ | n | TPOT vs OURS |
|---|---|---:|---:|---|---|---|---|---:|---:|
| ShareGPT | Qwen3 | 512 | 4 | OURS (HAQ) | 112.29 [106.29, 112.46] | 1.745 [1.731, 3.744] | 0.5509 [0.5501, 0.5527] | 6 | — |
| ShareGPT | Qwen3 | 512 | 4 | OURS-random | 101.32 [96.41, 101.73] | 1.741 [1.733, 3.698] | 0.6135 [0.6113, 0.6160] | 6 | +11.4% |
| ShareGPT | Qwen3 | 1024 | 4 | OURS (HAQ) | 109.62 [103.84, 110.77] | 2.512 [2.492, 4.399] | 0.5515 [0.5462, 0.5564] | 6 | — |
| ShareGPT | Qwen3 | 1024 | 4 | OURS-random | 99.33 [94.75, 100.26] | 2.517 [2.489, 4.409] | 0.6122 [0.6084, 0.6171] | 6 | +11.0% |
| ShareGPT | DeepSeekV2Lite | 512 | 4 | OURS (HAQ) | 189.06 [187.38, 190.58] | 0.728 [0.596, 0.871] | 0.3323 [0.3317, 0.3333] | 4 | — |
| ShareGPT | DeepSeekV2Lite | 512 | 4 | OURS-random | 164.27 [162.86, 164.87] | 0.747 [0.607, 0.861] | 0.3846 [0.3834, 0.3856] | 4 | +15.7% |
| ShareGPT | DeepSeekV2Lite | 1024 | 4 | OURS (HAQ) | 186.39 [184.41, 188.95] | 1.056 [0.951, 1.165] | 0.3320 [0.3290, 0.3341] | 4 | — |
| ShareGPT | DeepSeekV2Lite | 1024 | 4 | OURS-random | 162.25 [160.64, 163.62] | 1.058 [0.952, 1.173] | 0.3842 [0.3815, 0.3863] | 4 | +15.7% |
| LMSYS-Chat-1M | Qwen3 | 512 | 4 | OURS (HAQ) | 109.48 [104.06, 110.09] | 1.757 [1.746, 3.777] | 0.5647 [0.5627, 0.5666] | 6 | — |
| LMSYS-Chat-1M | Qwen3 | 512 | 4 | OURS-random | 97.53 [93.48, 98.39] | 1.758 [1.744, 3.760] | 0.6347 [0.6320, 0.6393] | 6 | +12.4% |
| LMSYS-Chat-1M | Qwen3 | 1024 | 4 | OURS (HAQ) | 107.31 [102.93, 108.41] | 2.506 [2.495, 4.432] | 0.5614 [0.5601, 0.5688] | 6 | — |
| LMSYS-Chat-1M | Qwen3 | 1024 | 4 | OURS-random | 97.31 [92.83, 97.96] | 2.533 [2.503, 4.400] | 0.6261 [0.6230, 0.6307] | 6 | +11.5% |
| LMSYS-Chat-1M | DeepSeekV2Lite | 512 | 4 | OURS (HAQ) | 188.30 [186.50, 190.06] | 0.730 [0.564, 0.871] | 0.3339 [0.3327, 0.3348] | 4 | — |
| LMSYS-Chat-1M | DeepSeekV2Lite | 512 | 4 | OURS-random | 163.78 [162.54, 164.51] | 0.733 [0.590, 0.874] | 0.3858 [0.3850, 0.3861] | 4 | +15.5% |
| LMSYS-Chat-1M | DeepSeekV2Lite | 1024 | 4 | OURS (HAQ) | 185.08 [184.59, 186.90] | 1.047 [0.931, 1.186] | 0.3335 [0.3331, 0.3357] | 4 | — |
| LMSYS-Chat-1M | DeepSeekV2Lite | 1024 | 4 | OURS-random | 161.49 [160.25, 162.52] | 1.065 [0.966, 1.163] | 0.3857 [0.3847, 0.3873] | 4 | +15.6% |

## Part B: Qwen3, ShareGPT, C30, B16, input 512, by GPU count (median [min, max], n)

| Dataset | Model | Input | GPUs | Method | TPS ↑ | TTFT (s) ↓ | TPOT (s/token) ↓ | n | TPOT vs OURS |
|---|---|---:|---:|---|---|---|---|---:|---:|
| ShareGPT | Qwen3 | 512 | 2 | OURS (HAQ) | 40.71 [39.35, 40.88] | 2.256 [2.239, 3.823] | 0.7618 [0.7577, 0.7655] | 6 | — |
| ShareGPT | Qwen3 | 512 | 2 | OURS-random | 38.58 [37.39, 38.86] | 2.274 [2.255, 3.823] | 0.8058 [0.8003, 0.8087] | 6 | +5.8% |
| ShareGPT | Qwen3 | 512 | 4 | OURS (HAQ) | 112.29 [106.29, 112.46] | 1.745 [1.731, 3.744] | 0.5509 [0.5501, 0.5527] | 6 | — |
| ShareGPT | Qwen3 | 512 | 4 | OURS-random | 101.32 [96.41, 101.73] | 1.741 [1.733, 3.698] | 0.6135 [0.6113, 0.6160] | 6 | +11.4% |
| ShareGPT | Qwen3 | 512 | 8 | OURS (HAQ) | 286.01 [270.08, 287.39] | 1.844 [1.807, 3.156] | 0.4258 [0.4233, 0.4313] | 6 | — |
| ShareGPT | Qwen3 | 512 | 8 | OURS-random | 252.25 [240.15, 253.26] | 1.822 [1.818, 3.134] | 0.4866 [0.4846, 0.4919] | 6 | +14.3% |

## Part C: R4 (GPUs 0/1/4/5), C30, B64, OURS-random only (median [min, max], n)

| Dataset | Model | Input | GPUs | Method | TPS ↑ | TTFT (s) ↓ | TPOT (s/token) ↓ | n | TPOT vs OURS |
|---|---|---:|---:|---|---|---|---|---:|---:|
| ShareGPT | Qwen3 | 512 | 4 | OURS-random | 319.78 [304.26, 322.80] | 4.162 [4.137, 5.534] | 0.7472 [0.7400, 0.7669] | 6 | — |
| ShareGPT | Qwen3 | 1024 | 4 | OURS-random | 301.30 [293.83, 301.64] | 7.643 [7.582, 8.481] | 0.7423 [0.7416, 0.7505] | 6 | — |
| ShareGPT | DeepSeekV2Lite | 512 | 4 | OURS-random | 598.49 [594.46, 601.57] | 2.059 [2.012, 2.118] | 0.4021 [0.3999, 0.4039] | 4 | — |
| ShareGPT | DeepSeekV2Lite | 1024 | 4 | OURS-random | 535.97 [528.52, 538.72] | 3.945 [3.843, 4.286] | 0.4229 [0.4212, 0.4240] | 4 | — |
| LMSYS-Chat-1M | Qwen3 | 512 | 4 | OURS-random | 308.89 [297.88, 310.43] | 4.189 [4.154, 5.542] | 0.7757 [0.7716, 0.7851] | 6 | — |
| LMSYS-Chat-1M | Qwen3 | 1024 | 4 | OURS-random | 288.27 [285.67, 289.96] | 7.842 [7.659, 8.325] | 0.7778 [0.7719, 0.7798] | 6 | — |
| LMSYS-Chat-1M | DeepSeekV2Lite | 512 | 4 | OURS-random | 598.65 [595.86, 601.09] | 2.106 [2.058, 2.173] | 0.4008 [0.3997, 0.4026] | 4 | — |
| LMSYS-Chat-1M | DeepSeekV2Lite | 1024 | 4 | OURS-random | 533.10 [521.03, 543.21] | 4.043 [3.825, 4.361] | 0.4237 [0.4180, 0.4299] | 4 | — |

## Notes

- **Reproduces the cache sweep:** R4 ShareGPT 512 matches `../cache_size_sweep_ablation` C30.

  | Model | This run (OURS / RANDOM) | Cache sweep C30 (OURS / RANDOM) |
  |---|---|---|
  | Qwen3 | 0.5509 / 0.6135 | 0.5510 / 0.6139 |
  | DeepSeek | 0.3323 / 0.3846 | 0.3353 / 0.3838 |

- **Interfered cell, rerun:** DeepSeek ShareGPT 512 was first run while another process used the
  machine.
  - Its first two jobs are kept outside this directory as
    `/home/hwlee/mgo-results/serial_ablation_l1024_scaling_20261011/interfered_ds_l512_r4_*`
    and are not counted.
  - The whole cell was rerun from HAQ r1 (`run_lmsys.log`, after the "relaunch" line).
- **Driver exit codes before 254c6bd were unreliable:** `$(date)` inside the echo reset `$?`, so
  every job logged rc=0. Each raw job here was checked for `status.json` status PASS instead.
- **Uncommitted working-tree change during `run_lmsys.sh` and `run_b64_random.sh`:** another session edited
  `scripts/env_offload_policy.py` at 10:05:40, before `run_lmsys.sh` started at 10:07:13. The
  edit was not committed here.
  - It replaces the hard-coded Stage-II near tolerance 200 with `MGO_NEAR_BPS` (default 200).
    `near_assignment_with_quota` already defaults to `near_bps=200`.
  - `MGO_NEAR_BPS` was unset in these runs, so the HAQ path is unchanged. RANDOM (code 19) is untouched.
  - The `source_commit` recorded by those 36 jobs therefore describes their behaviour.
- **TTFT maximums of 3–4.4 s** on Qwen3 come from the first target of each job. The other targets
  sit at the median. The median is the reported value, and the range keeps every sample.
- **Tokens differ between policies, so routes diverge.** This is the same caveat as the cache sweep:
  an end-to-end comparison on the same prompts, not a fixed-route one.
- **R2/R8 runner change:** `scripts/run_qwen_r8_job.py` gained `--decode-policy` and `--ours-policy BR`
  in f3d69b3 so it can run this configuration. R2/R8 raws are copied from
  `<manifest root>/jobs/sab_{haq,random}_full_v{1,2}`.

## Reproduce

- `run.sh` (ShareGPT 1024 both models, ShareGPT 512 Qwen3 R2/R4/R8) and `run_lmsys.sh`
  (DeepSeek ShareGPT 512, LMSYS both models and inputs) are the exact drivers as run.
  - Their logs are `run.log` and `run_lmsys.log`.
  - Output root: `/home/hwlee/mgo-results/serial_ablation_l1024_scaling_20261011`.
- `run_b64_random.sh` (Part C) is the B64 OURS-random driver, logged in `run_b64_random.log`.
- `python summarize.py` rebuilds `SUMMARY.json` and the tables from `raw/`.
