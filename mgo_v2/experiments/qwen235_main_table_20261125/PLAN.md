# Qwen3-235B-A22B main-table cell (R4, C30, ShareGPT, B16, L512, O64)

Owner request (2026-11-25): one main-table cell on Qwen3-235B-A22B-Instruct-2507, five systems, TPS / TTFT / TPOT,
at least three independent timed trials per system (five in a cell that is unstable), and a GPU memory
footprint at the HBM peak.

## Workload

- Physical GPUs 0/1/4/5 (H100 80GB HBM3, NVLink), world 4, rank-local batch 16 (64 global requests),
  input 512, output 64, greedy, no EOS stop, BF16.
- Manifest: `/home/hwlee/mgo-results/qwen235_main_table_20261125/WORKLOADS.json`, cell
  `Qwen3-235B_ShareGPT_R4_C30_B16_L512_O64`.
  - The 235B tokenizer (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`) is
    byte-identical to Qwen3-30B-A3B-Instruct-2507, so the frozen 30B main-table ShareGPT B16 L512 token IDs
    (warmup and target, same sha256) are reused unchanged.
- C30: 94 × 128 = 12032 routed experts × 30% = 3609 slots, per rank [903, 902, 902, 902], 36 MiB per
  expert, 136,235,188,224 B (126.9 GiB) in total.

## Systems

| System | Definition |
|---|---|
| DeepSpeed ZeRO-Inference | `headline_deepspeed_worker.py`, ZeRO-3 BF16 CPU parameter offload; live-parameter window scaled to the C30 budget as in the main table. |
| llama.cpp | `headline_llama_qwen235_sync_worker.py` + `headline_llama_qwen235_sync.cpp`, synchronous batch, 32 threads, CUDA graphs off, graph reuse off. C30 as whole expert layers: floor(902/128) = 7 layers per GPU (28 GPU, 66 CPU layers, 25 slots per GPU unused, reported). |
| MoE-Infinity (repaired) | `headline_infinity_worker.py`, same repaired build as the main table; Qwen treatment (SDPA, speculative EAM). New offload store for 235B. |
| main_OURS | main-table definition: native C++ experts, optimized prefill/decode paths, Near placement, BF16 partials, prefetch off, overlap mode, cold cache per repeat. |
| OURS-random (overlap) | main_OURS with prefill and decode policy RANDOM (code 19). |
| OURS-random (serial, definition B) | prefill and decode RANDOM with `--h2d-serial-ablation` (as in `ttft_random_prefill_b64_20261013`). |

### Approved deviation for OURS

A rank-private full pinned expert copy needs 4 × 423 GiB, which does not fit the 1.9 TB host. All OURS
variants use one shared pinned copy in `/dev/shm`, mapped and `cudaHostRegister`ed by every rank. The H2D
path stays direct pinned DMA.

## Trials

- Each job: one untimed warmup batch (disjoint requests) plus three timed target batches, cold expert
  cache / released parameters before each batch.
- If a system's three TPOT values spread by more than 5% (max/min − 1), run that system to five timed
  trials.

## Memory footprint (per GPU, at the HBM peak)

Dense model weights, GPU-resident routed expert weights, KV cache, runtime buffers and temporary workspace
(derived: peak allocated − dense − experts − KV), peak allocated, peak reserved, actual GPU memory usage
(nvidia-smi, sampled every second by the supervisor), and the expert-cache residency ratio
(GPU-resident routed experts / 12032).
