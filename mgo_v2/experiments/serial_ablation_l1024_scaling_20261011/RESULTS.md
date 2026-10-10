# Serial ablation: OURS (HAQ) vs OURS-random at input 1024, and Qwen3 GPU scaling

OURS (HAQ, code 16) and OURS-random (RANDOM, code 19) in the serial ablation mode of
`../cache_size_sweep_ablation`. All 20 jobs ran at commit f3d69b3 and passed.

**Main result:** RANDOM's TPOT is 11.0% higher than OURS on Qwen3 and 15.7% higher on DeepSeek
at input 1024, which matches the input-512 C30 gaps. On Qwen3 input 512 the gap grows with GPU
count: +5.8% (2 GPUs), +11.4% (4), +14.3% (8). TTFT is the same for both policies, as expected,
because prefill is BR in both and the policy switches at decode step 1.

## Setting

- **Common:** ShareGPT, C30 expert budget, rank-local B16, output 64, greedy, no EOS stop.
- **Serial ablation mode (same flags as `../cache_size_sweep_ablation/run_*.sh`):**
  native serial per-expert executor, `--h2d-serial-ablation`, prefetch off, BR prefill,
  decode policy switched at step 1, empty expert cache at each target.
- **Part A:** R4 (GPUs 0/1/4/5), input 1024, Qwen3-30B-A3B-Instruct-2507 and DeepSeek-V2-Lite-Chat.
  Manifest: the main-table `main_table_2x2_20261008/ShareGPT/WORKLOADS.json`.
- **Part B:** Qwen3, input 512, R2 (GPUs 0/1), R4 (0/1/4/5), R8 (0–7).
  Manifests: `qwen_r2_sharegpt_b16_l512_20261010`, `qwen_cache_ablation_20261009` (C30 cell),
  `qwen_r8_sharegpt_b16_l512_20261009`. R4 was rerun at this commit rather than reused.
- **Order:** each cell runs HAQ r1, RANDOM r1, RANDOM r2, HAQ r2. `MGO_RANDOM_SALT` = round.
- **Samples:** every target is kept (repeat 0 is the in-job warmup). Qwen3: 2 jobs × 3 targets = 6.
  DeepSeek: 2 jobs × 2 targets = 4.
- **TPS** = global requests × 64 / E2E, the main-table definition.

## Part A: R4, ShareGPT, C30, B16, input 1024 (median [min, max], n)

| Model | GPUs | Method | TPS ↑ | TTFT (s) ↓ | TPOT (s/token) ↓ | n | TPOT vs OURS |
|---|---:|---|---|---|---|---:|---:|
| Qwen3 | 4 | OURS (HAQ) | 109.62 [103.84, 110.77] | 2.512 [2.492, 4.399] | 0.5515 [0.5462, 0.5564] | 6 | — |
| Qwen3 | 4 | OURS-random | 99.33 [94.75, 100.26] | 2.517 [2.489, 4.409] | 0.6122 [0.6084, 0.6171] | 6 | +11.0% |
| DeepSeekV2Lite | 4 | OURS (HAQ) | 186.39 [184.41, 188.95] | 1.056 [0.951, 1.165] | 0.3320 [0.3290, 0.3341] | 4 | — |
| DeepSeekV2Lite | 4 | OURS-random | 162.25 [160.64, 163.62] | 1.058 [0.952, 1.173] | 0.3842 [0.3815, 0.3863] | 4 | +15.7% |

## Part B: Qwen3, ShareGPT, C30, B16, input 512 (median [min, max], n)

| Model | GPUs | Method | TPS ↑ | TTFT (s) ↓ | TPOT (s/token) ↓ | n | TPOT vs OURS |
|---|---:|---|---|---|---|---:|---:|
| Qwen3 | 2 | OURS (HAQ) | 40.71 [39.35, 40.88] | 2.256 [2.239, 3.823] | 0.7618 [0.7577, 0.7655] | 6 | — |
| Qwen3 | 2 | OURS-random | 38.58 [37.39, 38.86] | 2.274 [2.255, 3.823] | 0.8058 [0.8003, 0.8087] | 6 | +5.8% |
| Qwen3 | 4 | OURS (HAQ) | 112.29 [106.29, 112.46] | 1.745 [1.731, 3.744] | 0.5509 [0.5501, 0.5527] | 6 | — |
| Qwen3 | 4 | OURS-random | 101.32 [96.41, 101.73] | 1.741 [1.733, 3.698] | 0.6135 [0.6113, 0.6160] | 6 | +11.4% |
| Qwen3 | 8 | OURS (HAQ) | 286.01 [270.08, 287.39] | 1.844 [1.807, 3.156] | 0.4258 [0.4233, 0.4313] | 6 | — |
| Qwen3 | 8 | OURS-random | 252.25 [240.15, 253.26] | 1.822 [1.818, 3.134] | 0.4866 [0.4846, 0.4919] | 6 | +14.3% |

## Notes

- **R4 input 512 reproduces the cache sweep:** TPOT 0.5509 / 0.6135 (+11.4%) here versus
  0.5510 / 0.6139 (+11.4%) in `../cache_size_sweep_ablation` C30.
- **TTFT maximums of 3–4.4 s** on Qwen3 come from the first target of each job; the other targets
  sit at the median. The median is the reported value; the range keeps every sample.
- **Tokens differ between policies, so routes diverge.** Same caveat as the cache sweep: this is an
  end-to-end comparison on the same prompts, not a fixed-route one.
- The R2/R8 runner (`scripts/run_qwen_r8_job.py`) gained `--decode-policy` and `--ours-policy BR`
  in f3d69b3 so it can run this configuration. R2/R8 raws are copied from
  `<manifest root>/jobs/sab_{haq,random}_full_v{1,2}`.

## Reproduce

- `run.sh` is the exact driver as run (`run.log` is its log). Output root:
  `/home/hwlee/mgo-results/serial_ablation_l1024_scaling_20261011`.
- `python summarize.py` rebuilds `SUMMARY.json` and the tables from `raw/`.
