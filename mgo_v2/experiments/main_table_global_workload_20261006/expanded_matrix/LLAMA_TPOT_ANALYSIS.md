# Why synchronous llama.cpp decode is faster: source/receipt audit

2026-10-07; no new GPU profiler or changes to the running matrix.
Scope: completed R4/C30/local B16/input256/global64/output64 cell.
Selected triplet means: llama TTFT32.886756s, TPOT0.278327s, E2E50.421337s;
OURS TTFT1.588954s, TPOT0.767430s, E2E49.937026s.
These are descriptive selected means; all5 records remain in RESULTS.

## Verified execution differences

- Native synchronous runner completes all requests' prefill before decoding.
  Each of63 subsequent steps evaluates all64 sequences, calls llama_synchronize,
  performs finite-checked greedy selection and timestamps completion. There is
  no EOS early termination, output64 is verified. Repeat3 has64 stamps and63
  intervals: min0.234127s, median0.275783s, max0.397807s. No asynchronous-tail
  metric or division by request count explains the low TPOT.
- Same BF16 model, no low-bit weight quantization: run.log records BF16 expert
  tensors. Static GPU experts cover14/48 layers (15.75GiB globally), remaining34
  layers use host expert tensors. CPU attention/KV offload remains disabled.
- runner sets op_offload=false. ggml-backend.cpp selects the weight's backend
  for weight operations; moving host-weight operations to GPU is conditional
  on op_offload. Thus host experts compute on CPU, avoiding per-miss expert
  weight H2D streaming. Activations/results still cross devices; not zero-copy
  or zero-communication inference.
- CPU thread settings are n_threads32/n_threads_batch64; actual selected pool
  depends on llama's batched graph flag. GPU portions reuse CUDA graphs:
  actual run.log contains CUDA Graph reused/warmup complete records.
- OURS uses full-pinned host weights, dynamic expert GPU cache, placement and
  distributed expert dispatch. Pinning accelerates copying; it does not
  eliminate copies or execute experts on CPU. Repeat3 starts scheduler counters
  at zero; summed end counters across4 ranks report178725 expert copies and
  1686660710400 bytes=1570.825195GiB for prefill plus63 decode forwards combined.
  This is transfer volume, NOT additive critical-path time or decode-only volume.
- Native llama uses static layer assignment without OURS' dynamic admission/
  eviction/placement and distributed expert all-to-all path. Its CPU expert
  compute trades arithmetic throughput for avoided weight transfer and control.

## Interpretation and limits

Low-row-count decode can favor CPU expert compute when GPU weight streaming,
CPU dispatch/control and communication costs dominate the alternative. Native
C++ execution and GPU graph reuse are additional plausible contributors.
Large-token prefill amortizes GPU transfers and uses GPU matrix throughput much
better: llama's large TTFT reverses most of its decode advantage in E2E here.
The measured E2Es are nearly tied; this is not evidence that llama dominates
all metrics or larger batches.

The architectural differences and timer semantics above are established.
Exact shares of the ~0.489s TPOT gap attributable to expert H2D, CPU/GPU compute,
placement, communication and launch overhead are NOT measured by these clean
runs. No claimed percentage attribution or claim that H2D alone explains it.
Resource fairness is a common GPU expert budget, not identical CPU execution:
llama explicitly uses CPU expert arithmetic; report that baseline mode.

Sources: examples/headline_llama_sync.cpp, llama_sync_worker.py; installed
llama.cpp ggml/src/ggml-backend.cpp weight-backend scheduler and
src/llama-context.cpp graph_compute; raw directories
/home/hwlee/mgo-results/headline_r4_20261007/expanded_R4_C30_B16_L256_O64_llama_sync_v1
and /home/hwlee/mgo-results/headline_r4_20261007/expanded_R4_C30_B16_L256_O64_ours_v1.
