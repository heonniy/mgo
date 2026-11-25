# Qwen3-235B-A22B main-table cell: results

R4 (GPUs 0/1/4/5, H100 80GB), C30, ShareGPT, rank-local B16 (64 requests), input 512, output 64, greedy,
BF16. Setting and definitions: `PLAN.md`. All selected jobs passed; raw receipts are under
`/home/hwlee/mgo-results/qwen235_main_table_20261125/jobs`.

**Main result:**
- **Throughput:** main_OURS reaches 46.9 tok/s. That is 4.5× MoE-Infinity (10.5), 8.6× DeepSpeed (5.45)
  and 14× llama.cpp (3.33).
- **Placement policy:** with only the policy changed, OURS-random (overlap) is 12.3% slower per output
  token than main_OURS (TPOT 1.456 vs 1.296 s) and its TTFT is 5.4% higher.

## Timing (median [min, max], n timed trials)

| System | TPS ↑ | TTFT (s) ↓ | TPOT (s/token) ↓ | n |
|---|---:|---:|---:|---:|
| DeepSpeed ZeRO-Inference | 5.45 [5.43, 6.15] | 15.451 [15.067, 16.325] | 11.6867 [10.3197, 11.7381] | 5 |
| llama.cpp | 3.33 [3.32, 3.34] | 1022.055 [1020.903, 1023.322] | 3.3353 [3.2381, 3.3373] | 3 |
| MoE-Infinity (repaired) | 10.49 [10.35, 10.68] | 12.900 [12.603, 13.462] | 5.9936 [5.8716, 6.0844] | 3 |
| main_OURS | 46.85 [45.76, 46.89] | 5.761 [5.756, 7.660] | 1.2964 [1.2953, 1.2994] | 3 |
| OURS-random (overlap) | 41.86 [41.08, 41.88] | 6.074 [6.069, 8.002] | 1.4560 [1.4556, 1.4566] | 3 |
| OURS-random (serial, def. B) | 29.78 [29.34, 29.81] | 6.101 [6.086, 7.984] | 2.0860 [2.0844, 2.0890] | 3 |

- **TPS** = 64 requests × 64 tokens / E2E. Repeat 0 of every job is an untimed warmup on disjoint requests.
- **DeepSpeed** had five trials. Its first three TPOTs spread by more than 5% (11.69, 11.69, 10.32), so the
  pipeline added a second job with two more. Four of the five sit at 11.64–11.74 s.
- **Stability of the other cells:** every other cell's three TPOTs are within 3.6%.
- **Qwen3-30B row:** reference only. It is the 30B main-table cell rerun at this commit. Its TPOT (0.469 s)
  is 26% lower than the main-table value (0.637 s, commit 6433242), because main_OURS-path optimizations
  landed after the main table (e.g. 288e2af). The 235B OURS rows all use the same current code.

## GPU memory footprint (per GPU, HBM peak of the timed repeats, GiB; min–max over GPUs and repeats)


| System | Dense (GiB) | GPU-resident routed experts (GiB) | KV cache (GiB) | Runtime buffers / workspace (GiB) | Peak allocated (GiB) | Peak reserved (GiB) | nvidia-smi max (GiB) | Expert-cache residency |
|---|---|---|---|---|---|---|---|---|
| DeepSpeed ZeRO-Inference | 0.55–0.55 (gathered; persistent 0.0015) | 21.35–21.35 (gathered per MoE block, transient) | 1.65–1.65 | 21.92–21.93 | 45.47–45.47 | 48.67–48.67 | 51.88 | 5.05% of routed-expert bytes at peak (transient) |
| llama.cpp | 3.24–4.12 (layer-split, not replicated) | 31.50 (7 whole layers/GPU; 0.88 GiB budget unused) | 2.06–2.25 (buffer) | 0.16–0.30 (compute buffer) | n/a (no torch) | n/a (no torch) | 38.74 | 29.79% (28 x 128 / 12032, static) |
| MoE-Infinity (repaired) | 5.63–6.75 (native allocator; est. = smi - experts - torch reserved, incl. CUDA context) | 31.71–31.75 (EAM peak charged) | 1.61–1.68 | in torch peak (KV + activations); native buffers in dense estimate | 5.20–5.45 (torch only) | 12.75–14.40 (torch only) | 52.78 | 30.00% (resident bytes / all routed experts) |
| main_OURS | 14.90–14.90 | 31.64–31.68 (capacity 31.71–31.75) | 1.65–1.65 | 1.58–1.72 | 49.84–50.01 | 51.73–55.60 | 59.50 | 29.93% (3601/12032) |
| OURS-random (overlap) | 14.90–14.90 | 31.64–31.68 (capacity 31.71–31.75) | 1.65–1.65 | 1.68–1.86 | 49.94–50.12 | 53.20–56.33 | 60.28 | 29.93% (3601/12032) |
| OURS-random (serial, def. B) | 14.90–14.90 | 31.64–31.68 (capacity 31.71–31.75) | 1.65–1.65 | 1.68–1.86 | 49.94–50.12 | 53.20–56.33 | 60.28 | 29.93% (3601/12032) |
| Qwen3-30B main_OURS (reference) | 2.87–2.87 | 4.03–4.03 (capacity 4.04–4.05) | 0.84–0.84 | 0.80–0.89 | 8.56–8.65 | 9.89–10.79 | 14.04 | 29.87% (1835/6144) |

How each system's columns were filled:
- **OURS variants:** per-rank `memory_footprint` receipt.
  - Measured: dense weights, expert-cache slots × 36 MiB, and KV tensors (equal to the analytic
    2·94·4·128·2 B·16·575).
  - Workspace = peak allocated − dense − expert-cache capacity − KV (derived).
  - Residency = resident experts / all routed experts (3601/12032). The two arena slots per rank stay
    empty with prefetch off.
- **DeepSpeed ZeRO-Inference:** keeps no expert cache. ZeRO-3 gathers each module, so the routed-expert
  figure is the peak of gathered expert parameters (calibration pass, periodic full scans), not a
  resident cache.
  - Dense = all-parameter live peak − that expert peak.
  - Workspace = peak allocated − live parameters − KV. It is large: ZeRO-3 prefetch and gather buffers.
- **MoE-Infinity:** its expert cache and dense weights live in its native allocator, outside torch.
  - Torch peaks cover only KV and activations.
  - Routed experts = EAM peak-charged bytes per GPU.
  - Dense is an estimate, not a measurement: nvidia-smi max − experts − torch reserved. It includes the
    CUDA context and native staging.
- **llama.cpp:** no torch allocator. Figures are its CUDA buffers from the log.
  - The model buffer holds the GPU share of dense layers (layer-split, not replicated) plus 7 whole
    expert layers per GPU (31.5 GiB).
  - KV and compute buffers are listed separately.
  - 25 slots (0.88 GiB) of each GPU's C30 budget cannot hold a whole layer and stay unused.
- **nvidia-smi max:** sampled every second by the job supervisor over the whole job.

## Deviations and incidents

- **OURS expert source:** all three OURS variants use one shared pinned host copy of the experts
  (`/dev/shm`, 423 GiB, `cudaHostRegister` by every rank) instead of rank-private copies, which would need
  4 × 423 GiB.
  - The H2D path is unchanged (direct pinned DMA).
  - This was approved before the run. The 30B reference keeps the main-table rank-private source.
- **OURS-random definitions:**
  - Overlap = main_OURS with prefill and decode policy RANDOM (code 19), `MGO_RANDOM_SALT=1`.
  - Serial (definition B) = the same plus `--h2d-serial-ablation`. It mixes the serial-mode slowdown with
    the policy effect.
- **MoE-Infinity:**
  - The first smoke failed on a bug in my KV-byte accounting (a loop variable kept the KV alive), fixed
    in 6ff62d1.
  - The retried smoke hit a native segfault (exit −11) in warmup prefill inside `wait_dispatch_local`.
    This is the same intermittent fault recorded in `../qwen_cache_ablation_20261009/BASELINE_FAILURES.md`.
  - The timed job then passed on the first attempt with the unchanged configuration. Failed attempts are
    kept as `failed_*` and excluded.
- **llama.cpp C30:** whole expert layers only, floor(902/128) = 7 per GPU (28 GPU, 66 CPU layers).
  - The layer split {0–23, 24–47, 48–71, 72–93} was predicted from llama.cpp's split rule and confirmed
    by the placement audit.

## Reproduce

- `run_all.sh`: the chained pipeline.
- `infinity_retry.sh`: MoE-Infinity with same-config retries.
- `q30b_memory_after.sh`: the 30B reference.
- `watchdog.sh`, `health_check.sh` and `monitor.sh`: hang handling and checks.
- Logs: `/home/hwlee/mgo-results/qwen235_main_table_20261125/orchestrator.log`.
- `python summarize.py` rebuilds `SUMMARY.json` and these tables.
