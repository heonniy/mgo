# llama.cpp balanced 3/3/3/3 expert-placement validation

This is a **single-cell validation only** for `R4_C30_B32_L512_O64`
(global B128). It does not replace the main-table baseline until the result is
reviewed.

Fixed settings:
- physical GPUs 0,1,4,5 only
- 32/32 CPU threads with the existing deterministic affinity
- CUDA Graph OFF
- ordinary llama graph reuse OFF
- `op_offload=false`: CPU-resident experts execute on CPU
- BF16, synchronous global batch, output64

Only static expert placement changes. The normal llama layer split is left
untouched; no tensor-split override is introduced. GPU expert tensors are kept
for layers:

`[2,6,10, 14,18,22, 26,30,34, 38,42,46]`.

All other MoE expert tensors are overridden to CPU. The loader log is parsed for
the actual layer-to-CUDA-device ownership. The run fails closed unless these 12
GPU expert layers resolve to exactly **3/3/3/3** across the four logical CUDA
devices.

Each GPU therefore holds 3 complete expert layers = 384 experts =
3.375 GiB of static expert weight, below every R4 C30 per-rank slot budget
(458-461 slots). Global static expert residency is 12 layers = 1536 experts =
13.5 GiB, leaving part of the 16.198 GiB global C30 budget unused because the
placement granularity is a complete expert layer.

Run only two primary repeats after warmup. The purpose is to inspect TPOT and
confirm actual 3/3/3/3 placement before deciding whether to use this baseline
more broadly.
