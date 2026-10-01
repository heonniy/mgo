from __future__ import annotations

import torch

from .types import LocalExecPlan
from .communicator import ReceivedBatch


class LegacySlotExecutorAdapter:
    """Rank-local adapter for the controller-owned legacy C++ slot executor.

    One torchrun process should expose one logical local CUDA device to this
    adapter. Multi-rank communication belongs to torch.distributed, not the
    legacy RPC executor.
    """

    def __init__(
        self,
        dispatcher,
        capacity: int,
        expert_bytes: int,
        num_experts: int,
    ):
        self.dispatcher = dispatcher
        self.num_experts = num_experts
        self.dispatcher.init_slot_pool(capacity, expert_bytes)

    def execute(
        self,
        layer: int,
        batch: ReceivedBatch,
        plan: LocalExecPlan,
        is_decode: bool,
    ) -> torch.Tensor:
        n = batch.hidden_states.shape[0]
        device = batch.hidden_states.device
        mask = torch.zeros(
            (n, self.num_experts), dtype=torch.bool, device=device
        )
        weights = torch.zeros(
            (n, self.num_experts), dtype=batch.hidden_states.dtype, device=device
        )

        valid = batch.expert_ids >= 0
        if valid.any():
            row = torch.arange(n, device=device).unsqueeze(1).expand_as(valid)
            rows = row[valid]
            experts = batch.expert_ids[valid]
            vals = batch.expert_weights[valid]
            mask[rows, experts] = True
            weights.index_put_((rows, experts), vals, accumulate=True)

        self.dispatcher.set_inputs(
            batch.hidden_states, mask, weights, is_decode
        )
        self.dispatcher.submit_plan(0, plan.hit_ops, plan.miss_ops)
        return self.dispatcher.wait_layer_done()
