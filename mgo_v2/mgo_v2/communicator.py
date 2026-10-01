from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.distributed as dist


def warmup_collectives() -> None:
    """Establish NCCL collective and peer paths before pinning a large store.

    NCCL can initialize point-to-point connections lazily on the first
    all-to-all; an all-reduce alone does not exercise that path. Keep this
    outside loading and measured generation, with every rank participating.
    """
    world = dist.get_world_size()
    warm = torch.ones(world * 1024, device="cuda")
    dist.all_reduce(warm)
    dist.all_to_all_single(torch.empty_like(warm), warm)
    dist.all_gather([torch.empty_like(warm) for _ in range(world)], warm)
    torch.cuda.synchronize()


@dataclass
class ReceivedBatch:
    hidden_states: torch.Tensor
    origin_rank: torch.Tensor
    origin_index: torch.Tensor
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor
    source_counts: list[int]


@dataclass
class ExpertPartials:
    values: torch.Tensor
    token_indices: torch.Tensor
    expert_ids: torch.Tensor


def _exchange_counts(send_counts: list[int], device: torch.device, stats=None) -> list[int]:
    world = dist.get_world_size()
    local = torch.tensor(send_counts, dtype=torch.int64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world)]
    operation = lambda: dist.all_gather(gathered, local)
    if stats is None:
        operation()
    else:
        stats.call("counts", local.numel() * local.element_size() * (world - 1), operation)
    rank = dist.get_rank()
    return torch.stack(gathered)[:, rank].tolist()


def _all_to_all_varlen(
    send: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    stats=None,
    kind="payload",
) -> torch.Tensor:
    shape = (sum(recv_counts),) + tuple(send.shape[1:])
    recv = torch.empty(shape, dtype=send.dtype, device=send.device)
    operation = lambda: dist.all_to_all_single(
        recv,
        send.contiguous(),
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    if stats is None:
        operation()
    else:
        row_bytes = send.element_size()
        for dim in send.shape[1:]:
            row_bytes *= dim
        stats.call(kind, (sum(send_counts) - send_counts[dist.get_rank()]) * row_bytes, operation)
    return recv


def dispatch_tokens(
    hidden_states: torch.Tensor,
    effective_routes: list[dict[int, float]],
    owner_by_expert: Dict[int, int],
    top_k: int,
    stats=None,
) -> ReceivedBatch:
    """Send each local token at most once per destination rank.

    The packet contains all experts on that destination rank for the token, so
    multiple same-rank expert routes share one hidden-state transfer.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = hidden_states.device

    chunks_idx: list[list[int]] = [[] for _ in range(world)]
    chunks_eids: list[list[list[int]]] = [[] for _ in range(world)]
    chunks_w: list[list[list[float]]] = [[] for _ in range(world)]

    for t, routes in enumerate(effective_routes):
        by_rank: dict[int, list[tuple[int, float]]] = {}
        for expert, weight in routes.items():
            dst = owner_by_expert[expert]
            by_rank.setdefault(dst, []).append((expert, weight))
        for dst, pairs in by_rank.items():
            if len(pairs) > top_k:
                raise RuntimeError("merged routes exceed top_k")
            eids = [-1] * top_k
            weights = [0.0] * top_k
            for j, (expert, weight) in enumerate(sorted(pairs)):
                eids[j] = int(expert)
                weights[j] = float(weight)
            chunks_idx[dst].append(t)
            chunks_eids[dst].append(eids)
            chunks_w[dst].append(weights)

    send_counts = [len(x) for x in chunks_idx]
    recv_counts = _exchange_counts(send_counts, device, stats)

    def cat_or_empty(chunks, tail, dtype):
        flat = []
        for dst in range(world):
            flat.extend(chunks[dst])
        if not flat:
            return torch.empty((0,) + tail, dtype=dtype, device=device)
        if isinstance(flat[0], torch.Tensor):
            return torch.stack(flat).to(device=device, dtype=dtype)
        return torch.tensor(flat, dtype=dtype, device=device)

    send_idx = cat_or_empty(chunks_idx, (), torch.int64)
    send_hidden = hidden_states.index_select(0, send_idx)
    send_eids = cat_or_empty(chunks_eids, (top_k,), torch.int64)
    send_w = cat_or_empty(chunks_w, (top_k,), hidden_states.dtype)

    recv_hidden = _all_to_all_varlen(send_hidden, send_counts, recv_counts, stats, "dispatch_hidden")
    recv_idx = _all_to_all_varlen(send_idx, send_counts, recv_counts, stats, "dispatch_metadata")
    recv_eids = _all_to_all_varlen(send_eids, send_counts, recv_counts, stats, "dispatch_metadata")
    recv_w = _all_to_all_varlen(send_w, send_counts, recv_counts, stats, "dispatch_metadata")

    origin_rank = torch.cat(
        [
            torch.full((recv_counts[src],), src, dtype=torch.int64, device=device)
            for src in range(world)
        ],
        dim=0,
    ) if sum(recv_counts) else torch.empty(0, dtype=torch.int64, device=device)

    return ReceivedBatch(
        recv_hidden, origin_rank, recv_idx, recv_eids, recv_w, recv_counts
    )


def return_partials(
    partials: torch.Tensor | ExpertPartials,
    received: ReceivedBatch,
    local_token_count: int,
    stats=None,
) -> torch.Tensor:
    """Return weighted experts in native order, or legacy rank-summed outputs."""
    world = dist.get_world_size()
    exact = isinstance(partials, ExpertPartials)
    values = partials.values if exact else partials
    device = values.device
    if exact:
        # Send only real expert outputs, without top-k-sized zero padding.
        destinations = received.origin_rank[partials.token_indices]
        order = destinations.argsort(stable=True)
        send_counts = destinations.bincount(minlength=world).tolist()
        send_out = values[order]
        send_idx = received.origin_index[partials.token_indices[order]]
        send_experts = partials.expert_ids[order]
    else:
        # Dispatch already grouped these rows by their source rank.
        send_counts = received.source_counts
        send_out = values
        send_idx = received.origin_index
    recv_counts = _exchange_counts(send_counts, device, stats)

    recv_out = _all_to_all_varlen(send_out, send_counts, recv_counts, stats, "return_outputs")
    recv_idx = _all_to_all_varlen(send_idx, send_counts, recv_counts, stats, "return_metadata")

    final = torch.zeros(
        (local_token_count, values.shape[-1]),
        dtype=values.dtype,
        device=device,
    )
    if exact:
        recv_experts = _all_to_all_varlen(send_experts, send_counts, recv_counts, stats, "return_metadata")
        for expert in sorted(recv_experts.unique().tolist()):
            if expert < 0:
                continue
            rows = torch.where(recv_experts == expert)[0]
            final.index_add_(0, recv_idx[rows], recv_out[rows])
    elif recv_idx.numel():
        final.index_add_(0, recv_idx, recv_out)
    return final
