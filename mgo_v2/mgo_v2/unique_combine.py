"""One-writer BF16 row addition; caller must validate immutable CPU indices."""
import torch
import triton
import triton.language as tl


def validate_unique_layout(event, local_rows):
    received = sum(event['recv_counts'])
    for _, rows, _, _ in event['groups']:
        if len(set(rows)) != len(rows) or any(r < 0 or r >= received for r in rows):
            return False
    offset = 0
    for count in event['send_counts']:
        rows = event['send_idx'][offset:offset + count]
        if len(rows) != count or len(set(rows)) != count or any(r < 0 or r >= local_rows for r in rows):
            return False
        offset += count
    return offset == len(event['send_idx'])


@triton.jit
def _add_rows(dest, rows, values, N, H: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N * H
    row = tl.load(rows + offsets // H, mask, other=0)
    address = row * H + offsets % H
    previous = tl.load(dest + address, mask, other=0).to(tl.float32)
    value = tl.load(values + offsets, mask, other=0).to(tl.float32)
    tl.store(dest + address, previous + value, mask)


def add_unique_rows_(dest, rows, values):
    # No atomics: different invocations remain ordered on the current stream.
    # Duplicate rows within an invocation are forbidden by the CPU layout gate.
    assert dest.is_cuda and rows.is_cuda and values.is_cuda
    assert dest.device == rows.device == values.device
    assert dest.dtype == values.dtype == torch.bfloat16 and rows.dtype == torch.int64
    assert dest.ndim == values.ndim == 2 and rows.ndim == 1
    assert dest.is_contiguous() and rows.is_contiguous() and values.is_contiguous()
    assert values.shape == (rows.numel(), dest.shape[1])
    if rows.numel():
        _add_rows[(triton.cdiv(values.numel(), 256),)](dest, rows, values, rows.numel(), dest.shape[1], 256)
    return dest
