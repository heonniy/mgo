from __future__ import annotations

import torch
import os

from .types import LocalExecPlan
from .communicator import ReceivedBatch, ExpertPartials


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
        self.expected_fetches = 0
        self.native_numerics = os.environ.get("MOE_EP_NATIVE_NUMERICS", "1") == "1"
        if self.native_numerics and os.environ.get("MOE_EP_NATIVE_NUMERICS") != "1":
            raise RuntimeError("call pin_rank_before_cuda_import() to configure native numerics")
        self.dispatcher.init_slot_pool(capacity, expert_bytes)

    def execute(
        self,
        layer: int,
        batch: ReceivedBatch,
        plan: LocalExecPlan,
        is_decode: bool,
    ) -> torch.Tensor | ExpertPartials:
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
        if self.native_numerics:
            # Preserve one weighted output per effective expert. Summing on
            # the owner first changes BF16 rounding relative to native Qwen3.
            chunks, token_indices, expert_ids = [], [], []
            for expert, rows, values in self.dispatcher.wait_layer_partials():
                chunks.append(values)
                token_indices.append(rows)
                expert_ids.append(torch.full_like(rows, expert))
            output = ExpertPartials(
                torch.cat(chunks) if chunks else batch.hidden_states.new_empty((0, batch.hidden_states.shape[1])),
                torch.cat(token_indices) if token_indices else torch.empty(0, dtype=torch.int64, device=device),
                torch.cat(expert_ids) if expert_ids else torch.empty(0, dtype=torch.int64, device=device))
        else:
            output = self.dispatcher.wait_layer_done()
        self.expected_fetches += len(plan.miss_ops)
        return output

    def assert_cache_matches(self, rank_cache):
        expected = {(l, e, entry.slot) for (l, e), entry in rank_cache.entries.items()}
        actual = set(map(tuple, self.dispatcher.get_cached_slots(0)))
        if actual != expected:
            raise AssertionError(f"physical slot drift: missing={expected - actual}, extra={actual - expected}")
        stats = self.dispatcher.get_cache_stats().tolist()
        if stats[3] != self.expected_fetches:
            raise AssertionError(f"physical H2D fetches {stats[3]} != logical misses {self.expected_fetches}")
