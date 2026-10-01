# mgo_v2

Clean research runtime for multi-GPU MoE CPU offloading.

mgo_v2 owns the control plane and loads only the compiled `_store` extension
from `MoE-Infinity-EP-archer-coslot`. Its Python model loader never constructs
the legacy RPC executor, placement manager or cache/prefetch controllers.
The C++ slot data plane includes the stream, tensor-binding and native-numerics
fixes required by this runtime.

## Final research pipeline

Per global layer event:

1. collect router metadata from all ranks;
2. classify global expert Hit / Miss;
3. expert-level substitution:
   - any missed source expert with a route weight >= 0.20 is protected exactly;
   - low-importance missed experts first reuse active exact/protected anchors;
   - otherwise use the best safe resident expert with similarity >= 0.65;
   - one source expert maps to one target for all of its routes;
4. admit residual exact misses with a pluggable rank-placement policy;
5. evict with one of:
   - LRU,
   - gate-score W=128,
   - diversity-preserving W=128 / k=1 / lambda=2;
6. dispatch each token at most once per destination rank with NCCL all-to-all;
7. execute rank-local cached/fetched experts through the legacy fixed-slot executor;
8. return weighted expert outputs and sum in expert order at the token-origin rank.

Native BF16 numerical parity requires preserving both Qwen3's individual
rounding steps and its expert accumulation order. Inputs are still deduplicated
per destination rank; exact outputs carry one vector per effective expert,
without padded top-k vectors. This return traffic can exceed a rank-summed
approximate implementation. `MOE_EP_NATIVE_NUMERICS=0` retains the legacy fused
path for explicit diagnostics; it is not the validated native-exact default.

## Admission policies

Implemented policy interfaces include:

- balanced random baseline;
- greedy current communication;
- current + inter-layer path affinity;
- Hungarian current communication;
- Hungarian + same-layer co-activation;
- Hungarian + same-layer + inter-layer path;
- pair-swap refinement using exact deduplicated token->rank communication.

All primary policies support hard per-event balanced rank quotas.

## What is deprecated

Do not use these legacy mechanisms as the cache authority when mgo_v2 is active:

- DeviceMapManager random placement;
- ExpertPrefetcher cache replacement;
- old ExpertCache / ExpertPriorityScore;
- autonomous Archer sparse eviction;
- RPC-based distributed expert execution.

Only one controller may own residency.

## Running on a server

Use a BF16 Qwen3 MoE checkpoint and the `qwen3` optional dependencies. The
validated environment is PyTorch 2.11.0+cu128, Transformers 4.57.6, CUDA 12.8,
CUTLASS 3.5.1, and eight H100 80GB GPUs. Build the extension with
`scripts/build_slot_extension.sh` (`CUTLASS_DIR`, optional `PYTHON`, CUDA and
uuid development headers must be available). The store-only build still needs
CUTLASS because the legacy fused diagnostic path remains available.

Run `scripts/prepare_offload.py --model MODEL --output STORE` once before
launching workers. The BF16 store is read-only during execution, is bound to
checkpoint identity in a manifest, and uses the legacy little-endian x86-64
index format. Preparation refuses to overwrite an incomplete/nonempty store.
Do not let multiple workers prepare the same destination.

With this directory on `PYTHONPATH`, launch `examples/validate_slots.py` using
`torchrun --standalone --nproc_per_node=1`, then 4 and 8. Use separate output
directories. `examples/native_reference.py` saves native fixed-prompt receipts;
`examples/validate_model.py` compares valid-token router choices, weights,
MoE outputs and generated tokens and audits each resident tensor's physical slot address.

`examples/qwen3_smoke.py` uses the same model loader. Supply `--similarity` as
a finite `[L,E,E]` NPY, and `--affinity` as an NPZ containing `same_layer`
`[L,E,E]` and `path` `[L-1,E,E]` when the selected admission policy requires
them. Use `--exact --admission random --eviction lru` for no substitution.
Missing affinity tables cause an error rather than silently changing policies.

Call `pin_rank_before_cuda_import()` before importing CUDA libraries, then
`warmup_collectives()` after process-group initialization and before loading
the pinned expert store. Run full-model jobs sequentially on this server;
concurrent context teardown and large pinned allocations stalled a diagnostic. Strict NUMA mode verifies the kernel memory policy. A PCI NUMA value of -1 is accepted
only when the OS exposes exactly one memory node. Native execution reads
resident slots directly (`MOE_EP_SLOT_VIEWS=1`); `=0` is a copy baseline for
profiling. `reset_slot_pool(capacity)` is only for a drained experiment boundary
with a new empty Python controller, never for autonomous replacement.

## Validation and measurements

See [SERVER_VALIDATION_RESULTS.md](SERVER_VALIDATION_RESULTS.md) for tested
scope, evidence and remaining measurement qualifications. All correctness runs
use GPU attention and a single logical residency controller.

`examples/benchmark_model.py` accepts an explicit JSON cell manifest and
question workload. It records empty-cache TTFT, fixed-step TPOT/throughput,
expert metrics, controller time, fetch bytes, and a short numeric quality screen.
Left padding has explicit per-sequence position IDs; padding rows never enter
expert policies. Throughput uses the maximum whole-generation wall clock
across ranks, and prefill snapshots allow separate decode counters.
Optional collective instrumentation records submitted peer tensor payloads and
CUDA intervals, not NCCL wire bytes or isolated kernel times. Instrumented
timings must be labelled. Results are resumable only with matching code, inputs
and settings. `examples/benchmark_fabric.py` separately measures H2D and
all-to-all bandwidth; `scripts/summarize_slot_trace.py` audits Nsight SQLite
exports for actual expert transfers and H2D/GEMM overlap.

`scripts/run_server_matrix.py` writes the A/B/C and batch/cache manifests and
runs them sequentially. `scripts/summarize_server_matrix.py` requires every
rank and repeat, checks identical repeated decisions/tokens, verifies fetch and
input-dispatch byte accounting, and emits JSON/CSV tables. `--partial` is only
for progress inspection. Nsight diagnostics can use
`examples/profile_model_worker.py` in a fresh output directory with
`--capture-range=cudaProfilerApi`; `scripts/summarize_model_trace.py` joins its
per-cell ranges to physical H2D receipts and actual NCCL kernel durations.

Use the separately validated Nsight 2025.6.1 configuration:
`--trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-event-trace=false
--flush-on-cudaprofilerstop=false --capture-range=cudaProfilerApi
--capture-range-end=stop`. Export its `.nsys-rep` with `nsys export --type=sqlite`
and pass the original ablation receipt directory as `--baseline` to
`summarize_model_trace.py`; it requires all worker completion records, exact
range coverage and identical runtime/input fingerprints, policies and outputs.
Large trace export can take several minutes after generation finishes.
`MGO_DEBUG_PTRACE=1` is an opt-in for native stack attachment to the profiling
workers; SIGUSR1 dumps their Python stacks. Periodic dumps are off by default
and can be explicitly requested with `MGO_STACK_DUMP_SECONDS`. Normal benchmark
workers do not install these debug hooks.

`examples/quality_reference.py --batches 4 8 16 32` creates batch-matched
native controls with one checkpoint load. Final aggregation requires matching
batch size, token limit, workload and checkpoint identity. The optional
`plot_server_matrix.py` (Matplotlib) renders only a fully audited summary.

See MIGRATION.md for the legacy issues this replaces.
