# Qwen C20–C50 baseline completion

Use the frozen Qwen3-30B ShareGPT R4/local-B16/input512/output64 workload from
`/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json` on
physical GPUs 0/1/4/5. Complete MoE-Infinity-repaired, DeepSpeed
ZeRO-Inference and synchronous balanced llama.cpp at C20/C30/C40/C50 so
their cache curves can be compared with the completed main_OURS curve.

The existing C30 three-repeat baseline jobs used the same warmup and target
SHA-256, cache budget and worker settings as this frozen manifest. Verify
their raw receipts and reuse those records with explicit provenance. Run the
remaining C20/C40/C50 cells as guarded exclusive jobs. Preserve all target
repeats and full ranges; use three unfiltered targets for compatibility with
the C30 records and original Qwen sweep. Keep one GPU job at a time, at least
2 GiB free per owner GPU, host-memory guards, and restore owner inference
loads after each job. Preserve failed attempts and resume under new labels.

For MoE-Infinity, require repaired EAM and actual expert residency within the
global budget. For DeepSpeed, require the audited live-parameter cap on every
rank, and report that the cap also charges non-expert parameters. For llama.cpp,
keep 32 CPU threads with fixed affinity, synchronous batch, CUDA graphs OFF,
graph reuse OFF, GPU attention/KV and CPU expert fallback. Place equal counts
of full expert layers on each GPU: 2/3/4/6 layers per device at C20/30/40/50,
respectively. Audit actual tensor placement and resident expert bytes in every
cell; quantized whole-layer granularity may leave part of C20/C30/C40 budgets
unused. Do not present that partition as a dynamic expert cache.

Report TTFT, TPOT and E2E for all four systems by cache size, with C30 reuse
flagged and a clear distinction between model/rank budget and total HBM. Raw
prompts, generated tokens, logs and resource traces stay outside Git. Commit
validated aggregate receipts as the sweep progresses.
