"""Per-model expert weight byte size for the coslot slot pool.

The GPU slot pool allocates ``cap × per_expert_bytes`` per GPU, and the cap is
derived as ``floor(HBM × sparse_hbm_ratio / per_expert_bytes)``.  This value
MUST equal archer's actual per-expert node byte size (``InitSlotPool`` slot
stride); otherwise Python's cap and archer's slot pool drift — Python plans
more slots than the GPU budget holds (→ cudaMalloc OOM) or fewer (→ wasted HBM).
Resolving it here, and passing it explicitly to ``init_slot_pool``, keeps the
two sides locked to one source of truth.

Default = 3 weight matrices (gate / up / down), each
``moe_intermediate_size × hidden_size`` elements, in the model's dtype.  This
matches Qwen3-MoE, Mixtral, and DeepSeek dense-act experts.  Add a model entry
to ``_OVERRIDE`` only when the expert structure differs (fused gate-up, a shared
expert folded into the node, or a non-standard dtype).
"""
from __future__ import annotations

_DTYPE_BYTES = {
    "bfloat16": 2, "float16": 2, "half": 2,
    "float32": 4, "float": 4,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "float8": 1,
}

# Explicit per-model overrides keyed by HF ``model_type``.  Empty by default —
# the formula below covers the standard 3-matrix MoE expert.  Reference values:
#   Qwen3-235B-A22B : 3 × 1536 × 4096 × 2 = 37,748,736 B = 36.0 MiB
#   Qwen3-30B-A3B   : 3 ×  768 × 2048 × 2 =  9,437,184 B =  9.0 MiB
_OVERRIDE: dict[str, int] = {}


def _dtype_bytes(model_config) -> int:
    dt = getattr(model_config, "torch_dtype", None)
    s = (str(dt).split(".")[-1].lower() if dt is not None else "bfloat16")
    return _DTYPE_BYTES.get(s, 2)


def resolve_expert_bytes(model_config) -> int:
    """Per-expert GPU byte size for ``model_config``.

    Returns 0 if undeterminable (caller should fall back to archer's
    auto-detect by passing 0 to ``init_slot_pool`` and to an unlimited cap).
    """
    model_type = str(getattr(model_config, "model_type", "") or "")
    if model_type in _OVERRIDE:
        return int(_OVERRIDE[model_type])
    hidden = int(getattr(model_config, "hidden_size", 0) or 0)
    inter = int(getattr(
        model_config, "moe_intermediate_size",
        getattr(model_config, "intermediate_size", 0),
    ) or 0)
    if hidden <= 0 or inter <= 0:
        return 0
    return 3 * inter * hidden * _dtype_bytes(model_config)
