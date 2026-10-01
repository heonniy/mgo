from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.distributed as dist


@dataclass
class ReceivedBatch:
    hidden_states: torch.Tensor
    origin_rank: torch.Tensor
    origin_index: torch.Tensor
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor


def _exchange_counts(send_counts: list[int], device: torch.device) -> list[int]:
    world = dist.get_world_size()
    local = torch.tensor(send_counts, dtype=torch.int64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    rank = dist.get_rank()
    return [int(gathered[src][rank].item()) for src in range(world)]


def _all_to_all_varlen(
    send: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
) -> torch.Tensor:
    shape = (sum(recv_counts),) + tuple(send.shape[1:])
    recv = torch.empty(shape, dtype=send.dtype, device=send.device)
    dist.all_to_all_single(
        recv,
        send.contiguous(),
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    return recv


def dispatch_tokens(
    hidden_states: torch.Tensor,
    effective_routes: list[dict[int, float]],
    owner_by_expert: Dict[int, int],
    top_k: int,
) -> ReceivedBatch:
    """Send each local token at most once per destination rank.

    The packet contains all experts on that destination rank for the token, so
    multiple same-rank expert routes share one hidden-state transfer.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = hidden_states.device

    chunks_hidden: list[list[torch.Tensor]] = [[] for _ in range(world)]
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
            chunks_hidden[dst].append(hidden_states[t])
            chunks_idx[dst].append(t)
            chunks_eids[dst].append(eids)
            chunks_w[dst].append(weights)

    send_counts = [len(x) for x in chunks_idx]
    recv_counts = _exchange_counts(send_counts, device)

    def cat_or_empty(chunks, tail, dtype):
        flat = []
        for dst in range(world):
            flat.extend(chunks[dst])
        if not flat:
            return torch.empty((0,) + tail, dtype=dtype, device=device)
        if isinstance(flat[0], torch.Tensor):
            return torch.stack(flat).to(device=device, dtype=dtype)
        return torch.tensor(flat, dtype=dtype, device=device)

    send_hidden = cat_or_empty(
        chunks_hidden, (hidden_states.shape[1],), hidden_states.dtype
    )
    send_idx = cat_or_empty(chunks_idx, (), torch.int64)
    send_eids = cat_or_empty(chunks_eids, (top_k,), torch.int64)
    send_w = cat_or_empty(chunks_w, (top_k,), hidden_states.dtype)

    recv_hidden = _all_to_all_varlen(send_hidden, send_counts, recv_counts)
    recv_idx = _all_to_all_varlen(send_idx, send_counts, recv_counts)
    recv_eids = _all_to_all_varlen(send_eids, send_counts, recv_counts)
    recv_w = _all_to_all_varlen(send_w, send_counts, recv_counts)

    origin_rank = torch.cat(
        [
            torch.full((recv_counts[src],), src, dtype=torch.int64, device=device)
            for src in range(world)
        ],
        dim=0,
    ) if sum(recv_counts) else torch.empty(0, dtype=torch.int64, device=device)

    return ReceivedBatch(
        recv_hidden, origin_rank, recv_idx, recv_eids, recv_w
    )


def return_partials(
    partials: torch.Tensor,
    received: ReceivedBatch,
    local_token_count: int,
) -> torch.Tensor:
    """Return one partial output per received token to its origin and sum."""
    world = dist.get_world_size()
    device = partials.device

    by_rank_out: list[list[torch.Tensor]] = [[] for _ in range(world)]
    by_rank_idx: list[list[int]] = [[] for _ in range(world)]
    for i in range(partials.shape[0]):
        dst = int(received.origin_rank[i].item())
        by_rank_out[dst].append(partials[i])
        by_rank_idx[dst].append(int(received.origin_index[i].item()))

    send_counts = [len(x) for x in by_rank_idx]
    recv_counts = _exchange_counts(send_counts, device)

    flat_out = [x for r in range(world) for x in by_rank_out[r]]
    flat_idx = [x for r in range(world) for x in by_rank_idx[r]]
    send_out = (
        torch.stack(flat_out)
        if flat_out
        else torch.empty((0, partials.shape[1]), dtype=partials.dtype, device=device)
    )
    send_idx = torch.tensor(flat_idx, dtype=torch.int64, device=device)

    recv_out = _all_to_all_varlen(send_out, send_counts, recv_counts)
    recv_idx = _all_to_all_varlen(send_idx, send_counts, recv_counts)

    final = torch.zeros(
        (local_token_count, partials.shape[1]),
        dtype=partials.dtype,
        device=device,
    )
    if recv_idx.numel():
        final.index_add_(0, recv_idx, recv_out)
    return final
