# B4 implementation checkpoint

Scope: C30/local B128/R4 on physical GPUs 0,1,4,5, BF16, frozen decode64.
H0 and H1b remain unchanged defaults/references. H2 is explicit opt-in only.

H2 uses three dynamic Triton kernels for gate/up, fused SiLU/multiply, and down,
plus a routing-weight/scatter kernel. BF16 projection outputs and activation
match the existing compiled arithmetic; dot accumulation remains FP32 as in
BF16 GEMM. The return partial accumulation order and precision are unchanged.
Weights are read directly from the live slot arena. No private expert weights,
CUDA graphs, frozen-signature discovery, or future-event inspection are used.

At each current event, batch-query pending tickets under one scheduler lock,
execute all ready groups, and retain outputs in original group order. When
nothing is ready, wait for the first pending demand slot and re-query. One
completion event protects every slot consumed by a wave. No all-slot barrier.
Masked out-of-range row tiles do not execute GEMM. No fixed-M padded GEMM.

Workspace is bounded by global batch times top-k (4096 rows): 66 MiB per GPU
for input, gate/up, activation, down, and original-order output buffers, plus
small per-wave metadata/indices. Metadata is current-event-only. Peak allocator
usage will be reported with correctness and physical runs.

Backend smoke: PyTorch 2.11.0+cu128, Triton 3.6.0, H100 BF16. Tested variable
rows including 1,2,7,31,32,65,127,256,511; all sampled outputs bitwise equal to
the compiled reference. Full real-model correctness is a separate gate, pending.

Validation order: full64 correctness -> isolated grouped calibration -> separate
first8 diagnostics H1b/H2 -> clean C30 repeats -> C30 gate. C60 only on PASS.
