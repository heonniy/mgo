"""NVLink token routing — pack tokens by destination rank, all-to-all,
local compute, all-to-all back, scatter-combine.

Used by ``EPExpertExecutor.run_layer`` in the multi-rank path. The execution
target rank for an expert comes from the policy plan (placement for hits,
fetch target for misses) — this module is policy-agnostic.

Padded all_to_all is deliberately avoided in favour of uneven
``dist.all_to_all_single`` with explicit split sizes (one extra exchange to
agree on sizes, then one exchange per payload). This avoids ep_size^2 worst-
case padding waste.

Debug instrumentation: ``MOE_EP_DEBUG_A2A=1`` logs every collective issued
from this module to ``/tmp/moe_ep_a2a_rank{R}.log`` with seq counter, layer,
send/recv split sizes. Pair with ``MOE_EP_DEBUG_DETERMINISM=1`` from cache.view
to triangulate deadlocks: identical counts but missing collective on one
rank means a code-path divergence (e.g., the zero-send fallback).
"""
from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist


_DEBUG_A2A = os.environ.get("MOE_EP_DEBUG_A2A", "0") == "1"
# Heavy debug also prints to stderr (in addition to the per-rank file) so
# launch logs surface every collective without grep'ing tmp files.
_HEAVY = os.environ.get("MOE_EP_HEAVY_DEBUG", "0") == "1"

# 2026-05-28 fast-a2a (default on).  Two independent optimisations:
#   * derived counts: send/recv split sizes are computed from the controller's
#     already-gathered global demand matrix (per_rank) instead of a per-route
#     ``exchange_counts`` all_to_all + ``.cpu()`` sync.  Removes 1 collective
#     and 2 blocking device syncs (exchange + pack bincount) per route.
#   * fused payload: hidden|expert_idx|weights are byte-packed into ONE buffer
#     and moved with a single all_to_all_single instead of three.
# MOE_EP_VERIFY_COUNTS=1 cross-checks derived counts against exchange_counts /
# bincount and asserts equality (validation runs); default off (fast path).
_FAST_A2A = os.environ.get("MOE_EP_FAST_A2A", "1") == "1"
_VERIFY_COUNTS = os.environ.get("MOE_EP_VERIFY_COUNTS", "0") == "1"
_dbg_seq = [0]
_dbg_fh = [None]


def _heavy_log(tag: str, payload: dict) -> None:
    if not _HEAVY:
        return
    try:
        rank = dist.get_rank()
    except Exception:
        rank = -1
    parts = " ".join(f"{k}={v}" for k, v in payload.items())
    print(f"[router rank={rank}] {tag} {parts}", flush=True)


def _dbg_close() -> None:
    fh = _dbg_fh[0]
    if fh is not None:
        try:
            fh.flush()
            fh.close()
        except Exception:
            pass
        _dbg_fh[0] = None


atexit.register(_dbg_close)


def _dbg_log(rank: int, layer_id: int, op: str, payload: dict) -> None:
    if not _DEBUG_A2A:
        return
    if _dbg_fh[0] is None:
        _dbg_fh[0] = open(f"/tmp/moe_ep_a2a_rank{rank}.log", "w", buffering=1)
    _dbg_seq[0] += 1
    parts = " ".join(f"{k}={v}" for k, v in payload.items())
    _dbg_fh[0].write(f"#{_dbg_seq[0]:05d} L={layer_id} {op} {parts}\n")


@dataclass
class RouterPlan:
    """Pre-computed routing metadata returned by ``pack_tokens``."""
    # Per-entry indices into the original local batch (length K).
    token_idx: torch.Tensor          # [K] long
    expert_idx: torch.Tensor         # [K] long
    weights: torch.Tensor            # [K] same dtype as routing_weights
    send_hidden: torch.Tensor        # [K, H] reordered by dst rank
    # Per-dst-rank counts (length ep_size).
    send_counts: List[int]


def make_filtered_router_mask(
    router_mask: torch.Tensor,           # [N, E] bool
    allowed_expert_ids,                  # Iterable[int] of expert_id (not (layer, expert))
    num_experts: int,
) -> torch.Tensor:
    """Cooperative offloading: router_mask 를 allowed_expert_ids 만 남기게 필터링.

    hit-only / miss-only 두 batch 로 분리 dispatch 할 때 사용.  pack_tokens
    는 router_mask 가 0 인 expert 의 (token, expert) 페어를 자연 skip 하므로,
    expert_rank_table 의 다른 항목이 -1 여도 (dst_ranks>=0).all() assert 통과.
    """
    allowed_list = sorted(int(e) for e in allowed_expert_ids)
    out = torch.zeros_like(router_mask)
    if not allowed_list:
        return out
    device = router_mask.device
    idx = torch.tensor(allowed_list, dtype=torch.long, device=device)
    # router_mask[:, idx] 가 True 인 위치만 out 에 True
    out[:, idx] = router_mask[:, idx]
    return out


def build_expert_rank_table(
    expert_to_rank: Dict[Tuple[int, int], int],
    layer_id: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Materialise (layer, expert) -> rank as a flat tensor for index ops.

    Only entries with ``l == layer_id`` are placed — the SAME expert_id at
    a DIFFERENT layer maps to a DIFFERENT cache entry and must not pollute
    this layer's routing table.  Entries left at -1 are the assertion gate
    in ``pack_tokens`` (``(dst_ranks >= 0).all()``); any -1 means the
    controller's classify missed an expert that the router demanded.
    """
    table = torch.full(
        (num_experts,), fill_value=-1, dtype=torch.int64, device=device,
    )
    for (l, e), r in expert_to_rank.items():
        if l == layer_id:
            table[e] = r
    return table


def derive_split_counts(
    per_rank_count,                      # list[ep_size][num_experts] (Python ints)
    routing_map: Dict[Tuple[int, int], int],
    layer_id: int,
    my_rank: int,
    ep_size: int,
) -> Tuple[List[int], List[int]]:
    """Derive (send_counts, recv_counts) for one route WITHOUT a collective.

    Uses the controller's already-gathered global demand matrix
    ``per_rank_count[r][e]`` (= # of rank r's local tokens demanding expert e,
    from the Phase-1 NCCL all_gather) plus the deterministic ``routing_map``
    ((layer, expert) -> executing rank).  Every rank derives the same matrix,
    so:
      send_counts[s] = Σ_{e: route(e)==s}  per_rank[my_rank][e]
      recv_counts[r] = Σ_{e: route(e)==my_rank} per_rank[r][e]
    This is byte-identical to ``pack_tokens``'s ``bincount`` (send) and
    ``exchange_counts``'s all_to_all (recv) because both sides start from the
    same router_mask column sums and the same routing table.

    No GPU sync: ``per_rank_count`` is a plain Python nested list (caller does
    ``per_rank.tolist()`` once on the already-on-CPU demand tensor).
    """
    send = [0] * ep_size
    recv = [0] * ep_size
    my_row = per_rank_count[my_rank]
    for (l, e), r in routing_map.items():
        if l != layer_id:
            continue
        if 0 <= r < ep_size:
            send[r] += int(my_row[e])
            if r == my_rank:
                for src in range(ep_size):
                    recv[src] += int(per_rank_count[src][e])
    return send, recv


def pack_tokens(
    hidden_states: torch.Tensor,         # [N, H]
    router_mask: torch.Tensor,           # [N, E] bool
    routing_weights_mask: torch.Tensor,  # [N, E] float
    expert_rank_table: torch.Tensor,     # [E] int64 -> rank
    ep_size: int,
    precomp_send_counts: List[int] = None,  # fast path: skip bincount.cpu()
) -> RouterPlan:
    # Defensive dim checks — these would crash later with cryptic errors
    # inside torch.bincount / all_to_all_single if violated. Fail here with
    # a clear message.
    if router_mask.dim() != 2:
        raise RuntimeError(
            f"pack_tokens: router_mask must be 2-D [N, E], got "
            f"shape={tuple(router_mask.shape)}")
    if router_mask.shape[0] != hidden_states.shape[0]:
        raise RuntimeError(
            f"pack_tokens: N mismatch hidden={hidden_states.shape[0]} "
            f"router_mask N={router_mask.shape[0]}")
    if router_mask.shape != routing_weights_mask.shape:
        raise RuntimeError(
            f"pack_tokens: router_mask {tuple(router_mask.shape)} "
            f"!= routing_weights_mask {tuple(routing_weights_mask.shape)}")
    if expert_rank_table.shape[0] != router_mask.shape[1]:
        raise RuntimeError(
            f"pack_tokens: E mismatch router_mask E={router_mask.shape[1]} "
            f"expert_rank_table len={expert_rank_table.shape[0]}")

    # Find selected (token, expert) pairs.
    nz = router_mask.nonzero(as_tuple=False)              # [K, 2]
    token_idx = nz[:, 0].contiguous()                     # [K]
    expert_idx = nz[:, 1].contiguous()                    # [K]
    weights = routing_weights_mask[token_idx, expert_idx].contiguous()

    # Destination rank per entry.
    dst_ranks = expert_rank_table[expert_idx]             # [K]
    # The (dst_ranks >= 0).all() completeness check forces a .item() device
    # sync.  On the fast path the controller already guarantees routing
    # completeness (drift=0 verified at Phase 4), so skip it; keep it on the
    # legacy/verify path where it is the diagnostic gate.
    if precomp_send_counts is None or _VERIFY_COUNTS:
        if not (dst_ranks >= 0).all():
            missing_mask = dst_ranks < 0
            missing_exp = expert_idx[missing_mask].unique().cpu().tolist()
            raise RuntimeError(
                f"pack_tokens: expert_rank_table missing entries for "
                f"experts {missing_exp[:10]}{'…' if len(missing_exp) > 10 else ''} "
                f"({len(missing_exp)} total).  Controller's classify+placement "
                f"may have failed to install one of the demanded (layer, expert) "
                f"keys — check command_dispatcher fetch_failures counter."
            )

    # Sort by (dst_rank, expert_idx) for clean per-dst contiguous blocks.
    sort_keys = dst_ranks * (router_mask.size(1) + 1) + expert_idx
    sort_idx = sort_keys.argsort()
    token_idx = token_idx[sort_idx]
    expert_idx = expert_idx[sort_idx]
    weights = weights[sort_idx]
    dst_ranks = dst_ranks[sort_idx]
    send_hidden = hidden_states[token_idx].contiguous()   # [K, H]

    if precomp_send_counts is not None:
        # Fast path: split sizes derived from global demand — no bincount.cpu()
        # sync.  (Optionally cross-checked below under MOE_EP_VERIFY_COUNTS.)
        send_counts: List[int] = list(precomp_send_counts)
        if _VERIFY_COUNTS:
            bc = torch.bincount(dst_ranks, minlength=ep_size).cpu().tolist()[:ep_size]
            assert bc == send_counts, (
                f"pack_tokens VERIFY: derived send_counts {send_counts} != "
                f"bincount {bc}")
    else:
        # Per-rank counts via bincount (legacy path).
        send_counts_t = torch.bincount(dst_ranks, minlength=ep_size)
        send_counts = send_counts_t.cpu().tolist()
        if len(send_counts) > ep_size:
            if any(send_counts[ep_size:]):
                raise RuntimeError(
                    f"pack_tokens: dst_ranks contains rank out of range "
                    f"[0, {ep_size}). counts past ep_size = {send_counts[ep_size:]}")
            send_counts = send_counts[:ep_size]

    _heavy_log("pack", {
        "N": hidden_states.shape[0], "H": hidden_states.shape[1],
        "K": int(nz.shape[0]), "send_counts": send_counts,
    })

    return RouterPlan(
        token_idx=token_idx,
        expert_idx=expert_idx,
        weights=weights,
        send_hidden=send_hidden,
        send_counts=send_counts,
    )


def exchange_counts(send_counts: List[int], ep_group, layer_id: int = -1) -> List[int]:
    """Symmetric all_to_all of one int per pair — agrees on recv sizes."""
    ep_size = len(send_counts)
    device = torch.cuda.current_device()
    send = torch.tensor(send_counts, dtype=torch.int64, device=device)
    recv = torch.empty(ep_size, dtype=torch.int64, device=device)
    if _DEBUG_A2A:
        _dbg_log(dist.get_rank(group=ep_group), layer_id, "exchange_counts:pre",
                 {"send": send_counts})
    dist.all_to_all_single(recv, send, group=ep_group)
    out = recv.cpu().tolist()
    if _DEBUG_A2A:
        _dbg_log(dist.get_rank(group=ep_group), layer_id, "exchange_counts:post",
                 {"recv": out})
    return out


def _fused_route(
    plan: RouterPlan, recv_counts: List[int], K_recv: int, ep_group,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Byte-pack hidden|expert_idx|weights into one buffer and move with a
    SINGLE all_to_all_single, then unpack.  Exact (raw byte copy, no numeric
    change).  Replaces three collectives with one."""
    device = plan.send_hidden.device
    K = plan.send_hidden.size(0)
    H = plan.send_hidden.size(1)
    eh = plan.send_hidden.element_size()
    ee = plan.expert_idx.element_size()
    ew = plan.weights.element_size()
    hb = H * eh
    row = hb + ee + ew

    send_buf = torch.empty(K, row, dtype=torch.uint8, device=device)
    send_buf[:, 0:hb] = plan.send_hidden.contiguous().view(torch.uint8).view(K, hb)
    send_buf[:, hb:hb + ee] = (
        plan.expert_idx.contiguous().reshape(K, 1).view(torch.uint8))
    send_buf[:, hb + ee:row] = (
        plan.weights.contiguous().reshape(K, 1).view(torch.uint8))

    recv_buf = torch.empty(K_recv, row, dtype=torch.uint8, device=device)
    dist.all_to_all_single(
        recv_buf, send_buf,
        output_split_sizes=recv_counts,
        input_split_sizes=plan.send_counts,
        group=ep_group,
    )
    # Unpack by copying byte columns INTO freshly-allocated typed tensors, then
    # viewing the (contiguous, properly-aligned) destinations as bytes.  This
    # avoids viewing a column-slice of recv_buf directly as a wider dtype, which
    # fails when the row stride isn't a multiple of the dtype size
    # (e.g. row=4108 not divisible by 8 → "view Byte as Long" error).
    recv_hidden = torch.empty(K_recv, H, dtype=plan.send_hidden.dtype, device=device)
    recv_hidden.view(torch.uint8).view(K_recv, hb).copy_(recv_buf[:, 0:hb])
    recv_expert_idx = torch.empty(K_recv, dtype=plan.expert_idx.dtype, device=device)
    recv_expert_idx.view(torch.uint8).view(K_recv, ee).copy_(recv_buf[:, hb:hb + ee])
    recv_weights = torch.empty(K_recv, dtype=plan.weights.dtype, device=device)
    recv_weights.view(torch.uint8).view(K_recv, ew).copy_(recv_buf[:, hb + ee:row])
    return recv_hidden, recv_expert_idx, recv_weights


def route_tokens(
    plan: RouterPlan,
    ep_group,
    layer_id: int = -1,
    recv_counts: List[int] = None,
    fuse: bool = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Move (hidden / expert_idx / weights) to their destination ranks.

    Returns (recv_hidden, recv_expert_idx, recv_weights, recv_counts).

    Fast path (default, MOE_EP_FAST_A2A=1):
      * ``recv_counts`` is supplied by the caller (derived from the controller's
        global demand) → no ``exchange_counts`` collective + ``.cpu()`` sync.
      * payloads fused into one all_to_all (``fuse``).
    Legacy path: ``recv_counts=None`` → ``exchange_counts``; ``fuse=False`` →
    three separate collectives.
    """
    if fuse is None:
        fuse = _FAST_A2A
    if recv_counts is None:
        recv_counts = exchange_counts(plan.send_counts, ep_group, layer_id=layer_id)
    elif _VERIFY_COUNTS:
        xchg = exchange_counts(plan.send_counts, ep_group, layer_id=layer_id)
        assert xchg == recv_counts, (
            f"route_tokens VERIFY: derived recv_counts {recv_counts} != "
            f"exchanged {xchg} (L{layer_id})")
    K_recv = sum(recv_counts)
    H = plan.send_hidden.size(1)
    device = plan.send_hidden.device
    rank = dist.get_rank(group=ep_group)

    if _DEBUG_A2A:
        _dbg_log(rank, layer_id, "route:pre",
                 {"K_send": sum(plan.send_counts), "K_recv": K_recv,
                  "send": plan.send_counts, "recv": recv_counts, "fuse": fuse})

    if fuse:
        recv_hidden, recv_expert_idx, recv_weights = _fused_route(
            plan, recv_counts, K_recv, ep_group)
        if _DEBUG_A2A:
            _dbg_log(rank, layer_id, "route:post(fused)", {})
        return recv_hidden, recv_expert_idx, recv_weights, recv_counts

    recv_hidden = torch.empty(K_recv, H, dtype=plan.send_hidden.dtype, device=device)
    dist.all_to_all_single(
        recv_hidden, plan.send_hidden,
        output_split_sizes=recv_counts,
        input_split_sizes=plan.send_counts,
        group=ep_group,
    )
    recv_expert_idx = torch.empty(K_recv, dtype=plan.expert_idx.dtype, device=device)
    dist.all_to_all_single(
        recv_expert_idx, plan.expert_idx,
        output_split_sizes=recv_counts,
        input_split_sizes=plan.send_counts,
        group=ep_group,
    )
    recv_weights = torch.empty(K_recv, dtype=plan.weights.dtype, device=device)
    dist.all_to_all_single(
        recv_weights, plan.weights,
        output_split_sizes=recv_counts,
        input_split_sizes=plan.send_counts,
        group=ep_group,
    )
    if _DEBUG_A2A:
        _dbg_log(rank, layer_id, "route:post", {})
    return recv_hidden, recv_expert_idx, recv_weights, recv_counts


def return_outputs(
    local_output: torch.Tensor,            # [K_recv, H]
    recv_counts: List[int],
    send_counts: List[int],
    ep_group,
    layer_id: int = -1,
) -> torch.Tensor:
    """All-to-all back: shape becomes [K, H], aligned with plan.token_idx.

    Cross-stream safety: ``local_output`` is produced by archer's
    ExpertDispatcher on its own CUDA stream.  NCCL collectives launch on
    the caller's current stream — without an explicit cross-stream wait,
    NCCL may read partial data.  We use ``record_stream`` so the allocator
    knows the buffer is in use by the NCCL stream and won't free it; and
    ``contiguous()`` forces a copy that synchronises if the underlying
    storage was on a different stream.

    Empty-input guard: when ``K_recv == 0`` we rebuild ``local_output`` as
    a zero-row tensor backed by real storage so NCCL doesn't see a null
    pointer on the asymmetric side.
    """
    K = sum(send_counts)
    H = local_output.size(1)
    device = local_output.device

    # Empty-input guard: allocate non-null storage and slice back to [0, H].
    if local_output.numel() == 0:
        local_output = torch.zeros(
            1, H, dtype=local_output.dtype, device=device,
        )[:0]
    else:
        # Tell the caching allocator that the current stream will use this
        # buffer until NCCL completes.  Without record_stream, the allocator
        # could reuse local_output's memory immediately after archer's
        # stream finishes — before NCCL reads it from the current stream.
        # M5 fix (2026-05-27): no longer swallow the exception. record_stream
        # is required for cross-stream safety; if it raises we want to know
        # (likely a CPU tensor or torch API drift). Without the fail-fast,
        # archer's cross-stream output could be freed before NCCL reads it
        # and silently NaN.
        if not local_output.is_cuda:
            raise RuntimeError(
                f"return_outputs: local_output must be on CUDA for "
                f"cross-stream record_stream, got device={local_output.device}")
        local_output.record_stream(torch.cuda.current_stream())
    local_output = local_output.contiguous()

    send_back = torch.empty(K, H, dtype=local_output.dtype, device=device)
    if _DEBUG_A2A:
        rank = dist.get_rank(group=ep_group)
        _dbg_log(rank, layer_id, "return_outputs:pre",
                 {"K_back": K, "in_split (recv)": recv_counts,
                  "out_split (send)": send_counts,
                  "in_numel": local_output.numel(),
                  "out_numel": send_back.numel()})
    dist.all_to_all_single(
        send_back, local_output,
        output_split_sizes=send_counts,
        input_split_sizes=recv_counts,
        group=ep_group,
    )
    if _DEBUG_A2A:
        _dbg_log(rank, layer_id, "return_outputs:post", {})
    return send_back


def scatter_combine(
    send_back: torch.Tensor,               # [K, H], dispatcher output per entry
    token_idx: torch.Tensor,               # [K] original token indices
    N: int,                                # number of local tokens
    H: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Sum per-entry outputs back to per-token outputs ([N, H])."""
    final = torch.zeros(N, H, dtype=dtype, device=device)
    if send_back.numel() > 0:
        final.index_add_(0, token_idx, send_back.to(dtype))
    return final
