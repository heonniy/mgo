"""PriorityAggregator — EAMC-driven expert priority for multi-process EP+DP.

Brings upstream MoE-Infinity's ``expert_predictor`` into the multi-process
world.  Each rank holds its own predictor instance (cheap: shares the same
EAMC trace_collection on disk), runs predict() on ITS local sample(s),
then NCCL all_reduce SUMs the priority matrices so every rank sees a
byte-identical global priority view.  The controller's downstream policies
(placement, eviction, prefetch) consume this view to make consistent
decisions.

Architecture notes:
  * **Layer × expert uniqueness**: archer's tracer stores per-sample
    matrices of shape ``[num_layers, num_experts]``.  The cell at
    ``[l, e]`` records how many times expert ``e`` was demanded at layer
    ``l``.  Same ``e`` at different ``l`` is a different cell — i.e. the
    EAMC representation already respects the (layer, expert) key.
  * **predict()'s side effect**: archer's ``ExpertPredictor.predict``
    internally calls ``tracer.update_entry`` before computing the priority
    matrix.  When this aggregator runs, ``qwen_ep`` MUST skip its own
    update_entry call to avoid double-counting (see qwen_ep.py).
  * **Stride**: prediction is expensive (cosine similarity across the full
    trace_collection).  ``MOE_EP_EAMC_STRIDE=N`` runs the expensive
    find_most_similar only on every N-th layer; the cheap update_entry
    still fires every layer so the trace stays complete.  Off-stride
    layers fall back to ``None`` priority → LRU eviction.

When ``MOE_EP_EAMC_PRIORITY=0`` (default), aggregate returns ``None``
immediately and qwen_ep handles its own trace updates.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
import torch.distributed as dist


class PriorityAggregator:
    """Wraps an upstream ``ExpertPredictor`` instance + provides NCCL
    aggregation so all ranks share the same priority view.

    The predictor itself is created by archer (one instance per rank,
    sharing the loaded EAMC trace).  We hold a reference + drive it.
    """

    def __init__(
        self,
        expert_predictor,        # archer's ExpertPredictor (one per rank)
        num_layers: int,
        num_experts: int,
        ep_group,
        device: Optional[torch.device] = None,
    ):
        self.expert_predictor = expert_predictor
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.ep_group = ep_group
        self.device = device or torch.device("cuda")
        # Master toggle.  See module docstring.
        self._enabled = os.environ.get("MOE_EP_EAMC_PRIORITY", "0") == "1"
        # Stride for the expensive find_most_similar path.  Updating the
        # tracer is always per layer so the trace stays complete.
        self._stride = max(1, int(os.environ.get("MOE_EP_EAMC_STRIDE", "8") or 8))
        # Counters for visibility
        self.n_predicts = 0
        self.n_updates = 0
        self.n_skipped_stride = 0

    def enabled(self) -> bool:
        return self._enabled and self.expert_predictor is not None

    # ---- internal helpers ----

    def _update_trace(self, seq_ids, layer_id: int, expert_index_per_seq) -> None:
        """Cheap per-layer trace update.  Mirrors what predict() does
        before its find_most_similar call.

        Performance: convert the entire [B, S, top_k] tensor to a CPU
        numpy array ONCE (single GPU→CPU sync per layer) instead of per
        sample.  archer's ``update_entry`` does ``.flatten().tolist()`` on
        its argument which would force B separate syncs if we sliced GPU
        tensors per-sample.
        """
        tracer = getattr(self.expert_predictor, "tracer", None)
        if tracer is None:
            return
        # Single GPU→CPU sync.  If already CPU (numpy or list), no-op.
        try:
            if isinstance(expert_index_per_seq, torch.Tensor):
                arr = expert_index_per_seq.detach().cpu().numpy()
            else:
                arr = expert_index_per_seq
        except Exception:
            return
        for i, seq_id in enumerate(seq_ids):
            if seq_id is None:
                continue
            try:
                tracer.update_entry(seq_id, arr[i], layer_id)
                self.n_updates += 1
            except Exception:
                pass

    def _predict_matrix(self, seq_id, layer_id: int):
        """Cosine-similarity find + layer decay (the expensive parts of
        ExpertPredictor.predict, without re-calling update_entry).

        Returns a numpy array [num_layers, num_experts] or None on failure.
        """
        tracer = getattr(self.expert_predictor, "tracer", None)
        if tracer is None:
            return None
        try:
            current_entry = tracer.get_entry(seq_id)
            expert_matrix = tracer.find_most_similar(
                current_entry.matrix, layer_id,
            )
        except Exception:
            return None
        # Apply same decay as upstream predict.
        try:
            expert_matrix[:layer_id, :] = 0
            decay = self.expert_predictor.layer_decay_func
            for l in range(layer_id, self.num_layers):
                expert_matrix[l] = (
                    expert_matrix[l] + 1e-8
                ) * decay(l, layer_id, self.num_layers)
        except Exception:
            return None
        return expert_matrix

    # ---- public API ----

    def aggregate(
        self,
        seq_ids: Sequence[int] = (),
        layer_id: int = 0,
        expert_index_per_seq=None,
    ) -> Optional[torch.Tensor]:
        """Per-layer EAMC priority for placement/eviction decisions.

        Returns ``[num_layers, num_experts]`` float32 on ``self.device``,
        all_reduce-SUM across the EP group → byte-identical on every rank.
        Returns ``None`` when:
          * EAMC priority is disabled (env-deterministic across ranks)
          * this is a stride-skip layer (layer_id is same across ranks)

        DEADLOCK SAFETY (Round 5 fix):
        The early-return decision is based ONLY on env / layer_id, both of
        which are guaranteed identical across ranks.  Per-rank data
        (``seq_ids`` / ``expert_index_per_seq``) may be asymmetric — e.g.
        with batch_size < world_size some ranks have 0 local samples.
        Ranks WITHOUT local data still call all_reduce, contributing a
        zero matrix; otherwise we'd deadlock waiting for the missing rank.
        """
        if not self.enabled():
            return None

        has_local_data = expert_index_per_seq is not None and bool(seq_ids)

        # Stride gate — layer_id is same across ranks, so all-or-none.
        # Cheap trace update still fires on local data (no NCCL).
        if (layer_id % self._stride) != 0:
            if has_local_data:
                self._update_trace(seq_ids, layer_id, expert_index_per_seq)
            self.n_skipped_stride += 1
            return None

        # FROM HERE: all ranks are committed to the all_reduce below.
        # Ranks without local data contribute zeros.
        local_matrix = torch.zeros(
            (self.num_layers, self.num_experts),
            dtype=torch.float32, device=self.device,
        )
        if has_local_data:
            self._update_trace(seq_ids, layer_id, expert_index_per_seq)
            for i, seq_id in enumerate(seq_ids):
                if seq_id is None:
                    continue
                mat = self._predict_matrix(seq_id, layer_id)
                if mat is None:
                    continue
                try:
                    if isinstance(mat, torch.Tensor):
                        m = mat.to(device=self.device, dtype=torch.float32,
                                   non_blocking=True)
                    else:
                        m = torch.as_tensor(
                            mat, dtype=torch.float32, device=self.device,
                        )
                except Exception:
                    continue
                if m.shape != (self.num_layers, self.num_experts):
                    continue
                local_matrix += m
                self.n_predicts += 1

        # NCCL all_reduce SUM → global priority on all ranks.
        # MUST run on every rank when reaching this point — see deadlock note.
        if self.ep_group is not None and dist.is_initialized():
            dist.all_reduce(
                local_matrix, op=dist.ReduceOp.SUM, group=self.ep_group,
            )
        return local_matrix

    # ---- introspection ----

    def snapshot(self) -> dict:
        return {
            "n_predicts": self.n_predicts,
            "n_updates": self.n_updates,
            "n_skipped_stride": self.n_skipped_stride,
            "stride": self._stride,
            "enabled": self.enabled(),
        }
