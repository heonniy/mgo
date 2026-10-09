# D1 fixed-continuation and route-replay implementation

The DeepSeek worker has three diagnostic route modes. `none` preserves the
existing model path. `capture` records every selected expert, BF16 routing
weight and FP32 router-probability tensor after a model forward on a fixed
continuation. `replay` injects those tensors at the patched MoE boundary on
each target forward; the model still computes its usual gate probabilities,
which are then replaced by the recorded values. Both modes are excluded from
the headline main table.

The fixed continuation is the original C20 target repeat 1, all 64 tokens
for the same 64 ShareGPT requests. The worker records its own predicted
argmax but feeds the fixed reference ID to the next step. It validates every
rank's request IDs against the frozen workload and records the input-token
hash. Route capture is a separate diagnostic target after warmup. Its
compressed per-rank route files remain outside Git. Replay preloads all
route tensors to each owner GPU before target timing, checks their exact
shapes and ID bounds, and fails if any MoE event has an unexpected shape or
count. A route-file SHA-256 is recorded with every target run.

Controlled-route timing requires a C20/C50 alternating sequence of three
guarded, uninstrumented targets per arm. The same per-rank route-file hashes
and fixed-token manifest hash must appear in all six runs. Each target has
its own warmup and empty expert-cache reset. The physical GPU set remains
0/1/4/5; GPUs 2/3/6/7 are reserved. Compare H2D, expert-group counts,
remote packet bytes, final tokens and per-rank memory before interpreting
TPOT. The capture run is diagnostic and cannot replace any clean target.
