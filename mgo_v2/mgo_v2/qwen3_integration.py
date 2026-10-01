from __future__ import annotations

import types

import torch


def _qwen3_mgo_forward(self, hidden_states: torch.Tensor):
    batch, seq, hidden = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden)
    router_logits = self.gate(flat)

    # Use explicit fp32 softmax so full probabilities are available to the
    # W=128 gate history. top-k weights are normalized exactly over selected
    # experts when the model config requests it.
    probs = torch.softmax(router_logits.float(), dim=-1)
    top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
    if getattr(self, "norm_topk_prob", True):
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
    top_vals = top_vals.to(flat.dtype)

    out = self.mgo_runtime.forward_layer(
        layer=self.layer_id,
        hidden_states=flat,
        selected_experts=top_idx,
        routing_weights=top_vals,
        full_router_probs=probs,
        is_decode=(seq == 1),
    )
    return out.view(batch, seq, hidden).to(hidden_states.dtype), router_logits


def attach_qwen3_runtime(model, runtime) -> int:
    """Patch Qwen3 MoE blocks to use mgo_v2.

    Works with the MoE-Infinity fork's Qwen3MoEBlock and is intentionally
    conservative for upstream HF blocks: a module must expose gate/top_k and a
    layer_id. Returns the number of patched MoE blocks.
    """
    count = 0
    for module in model.modules():
        name = module.__class__.__name__.lower()
        if "qwen3" not in name or "moe" not in name:
            continue
        if not all(hasattr(module, x) for x in ("gate", "top_k", "layer_id")):
            continue
        if hasattr(module, "_mgo_v2_original_forward"):
            continue
        module._mgo_v2_original_forward = module.forward
        module.mgo_runtime = runtime
        module.forward = types.MethodType(_qwen3_mgo_forward, module)
        count += 1
    if count == 0:
        raise RuntimeError("no compatible Qwen3 MoE blocks found")
    return count


def detach_qwen3_runtime(model) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, "_mgo_v2_original_forward"):
            module.forward = module._mgo_v2_original_forward
            del module._mgo_v2_original_forward
            if hasattr(module, "mgo_runtime"):
                del module.mgo_runtime
            count += 1
    return count
