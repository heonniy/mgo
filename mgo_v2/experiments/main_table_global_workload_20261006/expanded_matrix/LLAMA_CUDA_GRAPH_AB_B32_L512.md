# llama.cpp CUDA Graph single-cell A/B

Scope is intentionally limited to one condition:

- R4 physical GPUs 0,1,4,5 only
- C30 expert budget
- local B32, therefore global B128 for llama.cpp
- input 512, output 64
- BF16 GGUF used by the headline runner
- 32/32 CPU threads with the same deterministic CPU affinity
- 14 GPU expert layers / 34 CPU expert layers
- `op_offload=false`
- ordinary llama computation-graph reuse ON

The only intended A/B variable is CUDA Graph capture/replay:

- ON: CUDA Graph support compiled and `GGML_CUDA_DISABLE_GRAPHS` absent.
- OFF: exact same binary, but `GGML_CUDA_DISABLE_GRAPHS=1`.

The pinned llama.cpp commit supports this runtime switch, so rebuilding separate
ON/OFF binaries is unnecessary and would introduce an avoidable confound.

Run exactly two primaries per arm after each arm's disjoint warmup. Do not select
or drop repeats. Require exact generated-token parity across corresponding ON/OFF
primaries and retain all per-step decode timings. Report TTFT, TPOT and E2E raw
values, means, OFF-ON deltas, and OFF/ON slowdown.

The revised main baseline is CUDA Graph OFF. The ON arm exists only to quantify
the timing impact in this single cell. No other batch/input/cache/thread setting
is part of this A/B.
