"""Qwen3MoeDecoderLayerEP — thin subclass of upstream ``Qwen3MoeDecoderLayer``.

History note (M2/M3 fix 2026-05-27):
  Earlier versions implemented a pipeline-overlap path gated on
  ``MOE_EP_PIPELINE_OVERLAP=1`` that pre-issued router + cache.sync.start_sync
  ahead of the MoE block and stashed an in-flight all_gather handle as
  ``mlp._precomputed_sync``.

  Two problems caused us to remove that path entirely:

  1. **Single-controller invariant violation**: ``cache.sync`` is legacy
     dead code (see auto-memory ``single_controller_path``). Re-importing
     ``start_sync`` from there reintroduced a second control loop alongside
     the GlobalCacheController + CommandDispatcher path — exactly the drift
     source the 2026-05-26 refactor eliminated. ``EPExpertExecutor`` no
     longer exposes a ``cache_view`` attribute, so ``_can_overlap()``
     always returned False and the overlap was non-functional anyway.

  2. **In-flight handle leak**: ``ep_executor`` consumed the precomputed
     tuple by ``del precomputed_sync`` without calling ``handle.work.wait()``,
     so the buffered all_gather op was discarded mid-flight. With PyTorch
     NCCL communicators, that risks reordering against the subsequent
     controller-driven all_gathers and the symptoms (NaN, byte-identity
     break, deadlock) would be silent and intermittent.

  Resolution: remove the overlap path. Keep the subclass as an identity
  shim so ``DistributedOffloadEngine.__enter__`` can still re-patch the
  decoder layer (e.g. for future model-aware customisations) without
  breaking the import chain.

  If/when cross-layer overlap is reintroduced, do it on the controller
  side (start an async demand all_gather inside ``handle_layer`` step 1
  with a wait inserted only at step 3 classify), not via a side-channel
  handle on the MLP block.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer as _UpstreamDecoderLayer,
)


class Qwen3MoeDecoderLayerEP(_UpstreamDecoderLayer):
    """Identity subclass — forwarding to the parent class is sufficient.

    Kept as a hook point: future model-aware decoration of the decoder layer
    (e.g. instrumentation around attn, KV-cache pinning) lands here without
    touching upstream code.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
