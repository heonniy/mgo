from __future__ import annotations

import types

import torch


def _qwen3_mgo_forward(self, hidden_states: torch.Tensor):
    batch, seq, hidden = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden)
    router_logits = self.gate(flat)

    # Preserve the legacy fork's actual router kernel when available so
    # mgo_v2 does not introduce an avoidable top-k numerical drift.  Full fp32
    # probabilities are computed separately only for W=128 gate-history state.
    probs = torch.softmax(router_logits.float(), dim=-1)
    if getattr(self, "lib", None) is not None and hasattr(self.lib, "topk_softmax"):
        router_mask, routing_weight_mask = self.lib.topk_softmax(router_logits)
        top_idx = torch.topk(router_logits, self.top_k, dim=-1).indices
        top_vals = routing_weight_mask.gather(1, top_idx).to(flat.dtype)
    else:
        top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
        if getattr(self, "norm_topk_prob", True):
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        top_vals = top_vals.to(flat.dtype)

    valid = getattr(self.mgo_runtime, "valid_token_indices", None)
    out = self.mgo_runtime.forward_layer(
        layer=self.layer_id,
        hidden_states=flat if valid is None else flat.index_select(0, valid),
        selected_experts=top_idx if valid is None else top_idx.index_select(0, valid),
        routing_weights=top_vals if valid is None else top_vals.index_select(0, valid),
        full_router_probs=probs if valid is None else probs.index_select(0, valid),
        is_decode=(seq == 1),
    )
    if valid is not None:
        out = torch.zeros_like(flat).index_copy_(0, valid, out)
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
    def capture_valid_tokens(_module, args, kwargs):
        mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        inputs = kwargs.get("input_ids", args[0] if args else None)
        if inputs is None:
            inputs = kwargs.get("inputs_embeds")
        runtime.valid_token_indices = None
        if mask is not None:
            if (mask.ndim != 2 or inputs is None or mask.shape[0] != inputs.shape[0]
                    or mask.shape[1] < inputs.shape[1]):
                raise ValueError("mgo_v2 requires a two-dimensional token attention mask")
            current = mask[:, -inputs.shape[1]:].reshape(-1).bool()
            if not bool(current.all()):
                runtime.valid_token_indices = current.nonzero().flatten()
    model._mgo_v2_mask_hook = model.register_forward_pre_hook(capture_valid_tokens, with_kwargs=True)
    return count


def detach_qwen3_runtime(model) -> int:
    if hasattr(model, "_mgo_v2_mask_hook"):
        model._mgo_v2_mask_hook.remove()
        del model._mgo_v2_mask_hook
    count = 0
    for module in model.modules():
        if hasattr(module, "_mgo_v2_original_forward"):
            module.forward = module._mgo_v2_original_forward
            del module._mgo_v2_original_forward
            if hasattr(module, "mgo_runtime"):
                del module.mgo_runtime
            count += 1
    return count
