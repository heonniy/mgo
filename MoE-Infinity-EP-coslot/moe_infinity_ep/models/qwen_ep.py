"""Qwen3MoEBlockEP — subclass of upstream Qwen3MoEBlock, overrides forward()
only. Keeps __init__ (so HF from_pretrained works identically) and inherits
identity for upstream's ``isinstance(module, Qwen3MoEBlock)`` setup loop at
``model_offload.py:612-659`` to bind expert_executor / lib / layer_id /
expert_tracer etc. onto our block too — no extra wiring needed.

The swap into HF's ``Qwen3MoeSparseMoeBlock`` slot is done in
``DistributedOffloadEngine.__enter__`` AFTER upstream's __enter__ has already
patched it to upstream's Qwen3MoEBlock.

Pipeline overlap REMOVED (M2/M3 fix 2026-05-27): the previous
``MOE_EP_PIPELINE_OVERLAP=1`` path consumed legacy ``cache.sync`` (dead code
under single-controller invariant) and silently leaked the in-flight
all_gather handle. See ``qwen_decoder_ep`` for the full rationale.
"""
from __future__ import annotations

import nvtx
import torch

from moe_infinity.models.qwen import Qwen3MoEBlock as _UpstreamQwen3MoEBlock


class Qwen3MoEBlockEP(_UpstreamQwen3MoEBlock):
    @nvtx.annotate("Qwen3MoEBlockEP", color="green")
    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        # Phase 8.4 — surface seq_id_list (already populated by harness) to
        # the executor so the GlobalCacheController.priority_aggregator can
        # run EAMC predict() per local sample.
        seq_id_list = getattr(self, "seq_id_list", None)
        if seq_id_list is not None:
            self.expert_executor._current_seq_ids = list(seq_id_list)
        else:
            self.expert_executor._current_seq_ids = []

        final_hidden_states, router_logits = self.expert_executor.run_layer(
            self.layer_id,
            flat_hidden,
            gate=self.gate,
            lib=self.lib,
            is_decode=(sequence_length == 1),
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        # EAMC feeding hook: if the harness has installed ``seq_id_list`` on
        # this block (one entry per batch row), feed archer's ExpertTracer
        # with the per-row top-k expert indices observed at this layer so a
        # subsequent ``save_eamc`` captures a populated trace_collection.
        #
        # Coordination with MOE_EP_EAMC_PRIORITY:
        # When priority is enabled, the controller's priority_aggregator
        # calls ``expert_predictor.predict()`` which ITSELF calls
        # ``tracer.update_entry`` internally — so an explicit update_entry
        # here would DOUBLE-COUNT the layer's contribution.  Skip when
        # priority is on; the predict call already populated trace.
        import os as _os
        seq_id_list = getattr(self, "seq_id_list", None)
        _eamc_priority_on = _os.environ.get("MOE_EP_EAMC_PRIORITY", "0") == "1"
        if seq_id_list is not None and not _eamc_priority_on:
            try:
                # router_logits shape: [batch_size*sequence_length, num_experts].
                # Reshape to [B, S, E] then top-k along expert dim.  Note:
                # ``(B, S, top_k)`` indices are PER-LAYER expert ids, and the
                # SAME expert_id at a DIFFERENT layer is a different weight.
                # tracer.update_entry stores them under ``(seq_id, layer_id)``
                # so the (layer, expert) uniqueness is preserved in EAMC.
                rl = router_logits.view(batch_size, sequence_length, -1)
                # M4 fix (2026-05-27): fallback chain `top_k → config.num_experts_per_tok
                # → 8` — the old `or rl.shape[-1]` defaulted to num_experts (128)
                # if both top_k and config were missing, which silently
                # poisoned the EAMC trace (every sample marked all experts).
                top_k = getattr(self, "top_k", None)
                if not top_k:
                    cfg = getattr(self, "config", None)
                    top_k = (getattr(cfg, "num_experts_per_tok", None)
                             if cfg is not None else None)
                if not top_k:
                    top_k = 8
                if top_k > rl.shape[-1]:
                    top_k = rl.shape[-1]
                _, idx = torch.topk(rl, k=int(top_k), dim=-1)
                idx_cpu = idx.detach().cpu().numpy()
                tracer = getattr(self, "expert_tracer", None)
                if tracer is not None:
                    for i, seq_id in enumerate(seq_id_list):
                        if seq_id is None:
                            continue
                        tracer.update_entry(seq_id, idx_cpu[i], self.layer_id)
            except Exception as _e:
                # EAMC population is best-effort; harness verifies via
                # eamc_io.verify_eamc() before saving anyway. Log first 5
                # exceptions so silent rot is visible.
                import os as _os2
                if (int(_os2.environ.get("MOE_EP_EAMC_LOG_ERRORS", "1") or 0)
                        and getattr(self, "_eamc_err_count", 0) < 5):
                    print(f"[qwen_ep] EAMC update_entry L{self.layer_id} "
                          f"err: {_e!r}", flush=True)
                    self._eamc_err_count = getattr(
                        self, "_eamc_err_count", 0) + 1

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)
        return final_hidden_states, router_logits
