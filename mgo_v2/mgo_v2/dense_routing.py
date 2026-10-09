"""Decode routing weights without a 128-column zero/scatter/convert chain."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_route_kernel(selected, weights, targets, output,
                        TOPK: tl.constexpr, EXPERTS: tl.constexpr):
    token = tl.program_id(0)
    expert = tl.arange(0, EXPERTS)
    score = tl.full((EXPERTS,), 0, tl.float32)
    for column in range(TOPK):
        original = tl.load(selected + token * TOPK + column)
        destination = tl.load(targets + original)
        weight = tl.load(weights + token * TOPK + column).to(tl.float32)
        score += tl.where(expert == destination, weight, 0.)
    tl.store(output + token * EXPERTS + expert, score)


def dense_decode_weights(selected, weights, targets, output=None):
    """Match the FP32 accumulation and final BF16 conversion of scatter_add_."""
    if selected.ndim != 2 or weights.shape != selected.shape:
        raise ValueError('expected matching [tokens, top-k] routing tensors')
    if targets.shape != (128,) or selected.shape[1] != 8:
        raise ValueError('expected Qwen3 128-expert top-8 decode routing')
    if not (selected.is_cuda and weights.is_cuda and targets.is_cuda):
        raise ValueError('routing tensors must be on CUDA')
    if not (selected.is_contiguous() and weights.is_contiguous() and targets.is_contiguous()):
        raise ValueError('routing tensors must be contiguous')
    if output is None:
        output = torch.empty((selected.shape[0], 128), dtype=weights.dtype, device=weights.device)
    if output.shape != (selected.shape[0], 128) or output.dtype != weights.dtype or output.device != weights.device:
        raise ValueError('invalid dense routing output buffer')
    _dense_route_kernel[(selected.shape[0],)](selected, weights, targets, output, 8, 128)
    return output
