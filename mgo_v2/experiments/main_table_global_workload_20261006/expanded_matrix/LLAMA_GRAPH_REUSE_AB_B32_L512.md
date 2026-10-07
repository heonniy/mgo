# llama.cpp ordinary graph-reuse check: B32/L512 only

The revised llama.cpp baseline disables **both** graph acceleration mechanisms:

- CUDA Graph capture/replay: OFF via `GGML_CUDA_DISABLE_GRAPHS=1`
- llama ordinary computation-graph reuse: OFF via `LLAMA_GRAPH_REUSE_DISABLE=1`

All other headline settings remain fixed: R4 GPUs 0,1,4,5; C30; local B32
(global B128); input512/output64; 32/32 CPU threads; fixed CPU affinity;
14 GPU expert layers / 34 CPU expert layers; `op_offload=false`.

The previous `9917b9c` CUDA-OFF arm kept ordinary graph reuse ON and measured
TPOT 0.741091 / 0.741742 s. It is retained as the reference. To isolate ordinary
graph reuse, run only one new arm with both mechanisms OFF, two primaries after
warmup, using the same binary. Require exact output-token parity.

This is the only requested graph-reuse timing check. Do not expand it to other
cells unless the owner explicitly asks.
