# llama.cpp synchronous baseline resource audit

Owner amendment, 2026-10-07. This audit exists because the synchronous llama.cpp TPOT is unexpectedly strong relative to prior reports.

## Frozen main-table policy

The main expanded matrix uses **32/32 CPU threads** for llama.cpp: `n_threads=32` and `n_threads_batch=32`. This choice is fixed before the new thread audit results are observed. Ordinary llama graph reuse remains enabled, but **CUDA Graph capture/replay is disabled at build time with `GGML_CUDA_GRAPHS=OFF`** for the revised baseline. Native MoE `MUL_MAT_ID`, weight repacking, split-mode=layer, and KQV/KV GPU offload remain enabled. This isolates CUDA-Graph acceleration without disabling the normal llama execution graph or native kernels.

C30/C60 expert residency remains unchanged: 14/28 complete GPU expert layers respectively, with remaining expert tensors explicitly overridden to CPU. `op_offload=false` is retained so CPU-resident expert operations are not opportunistically executed on GPU.

## CPU-budget control

All llama CPU work is bound to a deterministic CPU set inherited by the llama threadpool. For a total budget of N=16/32/64 threads, take N/4 CPUs from each of the existing fixed GPU-local CPU pools for physical GPUs 0,1,4,5. This keeps the experiment balanced across the same host/NUMA footprint used by the four-rank experiments and prevents the single llama process from wandering over unrestricted host CPUs.

The audit sweeps **16/16, 32/32, 64/64** on R4/C30/B64/L512/O64 by default. Thread count is an explicit required argument; synchronous llama jobs fail closed if it is omitted.

## Correctness and implementation guards

The custom runner installs a llama log callback during model loading and requires exact CPU expert override evidence. Qwen3-30B-A3B has 144 expert tensors (3 per layer x48). For C30 it must observe exactly 102 CPU expert tensor overrides in layers 0-33 and infer 42 GPU expert tensors in the remaining 14 layers. C60 analogously requires 60 CPU and 84 GPU expert tensors. The byte accounting remains 9 MiB/expert x128 experts/layer.

Every result records `n_batch=2048`, `n_ubatch=512`, prefill token count/call count, 63 synchronous decode calls, decode tokens per call, both CPU thread fields, placement settings, and every per-step decode duration. The Python worker recomputes TPOT from those 63 step intervals and rejects any discrepancy.

The 16/32/64 audit requires exact generated-token parity for every corresponding primary repeat across thread counts. It also requires placement audit PASS at every point. No timing sample is dropped.

## Build/provenance guard

The worker now checks both source SHA256 and binary SHA256 against `LLAMA_BUILD.json`. Any source edit therefore blocks execution until `scripts/build_headline_llama_sync.py` rebuilds the native runner and refreshes the receipt. The build script reconfigures the pinned llama.cpp tree with `-DGGML_CUDA_GRAPHS=OFF`, rebuilds the backend, and refuses to write an eligible receipt unless CMakeCache reports CUDA Graphs OFF and CUDA itself ON. The runtime worker also rejects any stale ON receipt. Ordinary llama graph reuse is explicitly kept enabled with `LLAMA_GRAPH_REUSE_DISABLE=0`.

## Scope

This commit prepares and validates the audit logic statically; it does **not** claim that 16/32/64 GPU runs have been executed. Run `scripts/run_llama_sync_audit.py --build` on the authorized server. Only physical GPUs 0,1,4,5 are permitted; never touch 2,3,6,7.
