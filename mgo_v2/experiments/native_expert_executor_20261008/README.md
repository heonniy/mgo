# Native expert execution

The implementation is an opt-in decode executor, not an import of the entire
legacy coslot engine. `--expert-executor native` selects it in
`headline_ours_worker.py` and `scripts/run_headline_job.py`. `h0` remains the
reference/default. B64 frozen64 validation is complete (see RESULTS.md);
unrestricted greedy and prefetch ON were not compared. Prefill remains H0.

It uses one C++ call for a currently ready subset, loops over ordinary
per-expert GEMMs in C++, reads existing GPU cache weights directly, and
records one shared slot-use event after the subset. No grouped GEMM, graph
cache, extra GPU stream, weight D2D copy, or global barrier is added.
If nothing is ready it retains H0's host submission wait plus compute-stream
event dependency, without a new host-DMA completion wait. It never waits to
fill a group. `NativeExpertExecutor(max_experts=...)` bounds the ready subset.

Build requirements: matching PyTorch/CUDA toolkit, C++ compiler and Ninja.
The tested host uses CUDA12.8 / PyTorch2.11 / H100 sm90. Ninja1.13.2 is
installed separately under `/home/hwlee/mgo-tools/native-expert-build`.
Before launching, set:

```sh
export PATH=/home/hwlee/mgo-tools/native-expert-build/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=1
```

Compilation is outside measurement, cached by PyTorch. Do not compile a
new extension concurrently in measured runs. Existing supervisor stops and
restores only owned loads on GPUs0/1/4/5 and applies shared-host memory guards.

## Reference inspected

`origin/main` at `27edf65`:

- `MoE-Infinity-EP-coslot/moe_infinity_ep/runtime/ep_executor.py`:
  single forward/return exchange and controller-owned slot execution.
- `MoE-Infinity-EP-archer-coslot/core/parallel/expert_dispatcher.cpp`,
  `GPUExecFunc`: verifies slot identity, waits for fetch dependency, gathers
  input, executes FFN, stores weighted partials, records last-use event.

The legacy code also performs slot-to-parameter copies and per-expert stream
completion waits. Those are deliberately not copied into the current overlap
runtime. This migration is a new small extension following the dependency
contract, not a claim that legacy engine performance carries over unchanged.

## Measurement boundaries

`micro_v1` preserves all four-rank checks and synthetic timings. Compare H0,
C++ called once per expert, and C++ called once per32 ready experts. The
single-vs-wave pair isolates crossing/loop batching more closely; H0-vs-C++
also changes FFN packaging and torch.compile dispatch. It cannot attribute
all improvement solely to the Python interpreter.

The full-model worker captures one H0 trajectory outside primary timing,
then fixes current-event routing IDs/weights/history and teacher tokens for
both executors. Same cold cache, BR seed, MAIN/PREFETCH capacities, H2D/peer
bytes and final cache roles are asserted. It measures64 decode forwards in
H0/native/native/H0 order, with no detailed timers in primary runs. This is
controlled executor timing, not unrestricted native-greedy generation.
