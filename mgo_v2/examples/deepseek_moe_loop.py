"""Preserve DeepSeek-V2 expert math with ZeRO-3-safe expert call order.

The stock loop skips experts with zero local tokens. Under ZeRO-3, each expert
call fetches partitioned parameters through a collective: different local
expert sets cause mismatched rank collective sequences. Use one rank union of
active experts, invoke that ordered union on every rank (including zero-row
calls), and keep the original local outputs and combine order.
"""
from types import MethodType

import torch
import torch.distributed as dist


def compact_moe(self, hidden_states, topk_ids, topk_weight):
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    local_counts = cnts.sum(dim=0)
    if dist.is_available() and dist.is_initialized():
        global_counts = local_counts.clone()
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        counts, global_counts = torch.stack((local_counts, global_counts)).cpu().tolist()
    else:
        counts = local_counts.cpu().tolist()
        global_counts = counts
    indices = topk_ids.view(-1).argsort()
    sorted_tokens = hidden_states[indices // topk_ids.shape[1]]

    outputs = []
    start_idx = 0
    for i, global_tokens in enumerate(global_counts):
        if global_tokens == 0:
            continue
        num_tokens = counts[i]
        end_idx = start_idx + num_tokens
        expert = self.experts[i + self.ep_rank * self.experts_per_rank]
        expert_out = expert(sorted_tokens[start_idx:end_idx])
        if num_tokens:
            outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[indices] = outs
    return (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def install_compact_deepseek_moe(model):
    count = 0
    for module in model.modules():
        if type(module).__name__ == 'DeepseekV2MoE':
            module.moe = MethodType(compact_moe, module)
            count += 1
    return count
