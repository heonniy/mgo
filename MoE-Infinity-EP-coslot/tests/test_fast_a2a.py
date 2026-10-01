"""Unit tests for the 2026-05-28 fast-a2a optimisations (no CUDA / no NCCL).

Covers the two pure pieces introduced this round:

  #1 derive_split_counts — derives all_to_all send/recv split sizes from the
     controller's global demand matrix (per_rank) + the routing map, replacing
     the per-route ``exchange_counts`` collective + ``.cpu()`` syncs.  Tested
     against a ground-truth packing of synthetic tokens (the values it MUST
     match are exactly what ``bincount(dst_ranks)`` and the all_to_all transpose
     would produce).

  #2 _fused_route — byte-packs hidden|expert_idx|weights into one buffer and
     moves it with a single all_to_all_single, then unpacks.  Tested for
     byte-exact round-trip (incl. the row-size-not-multiple-of-8 case that the
     naive ``.view(int64)`` unpack tripped on) by monkeypatching
     ``dist.all_to_all_single`` to an identity copy.
"""
from __future__ import annotations

import torch

from moe_infinity_ep.exec import nvlink_router
from moe_infinity_ep.exec.nvlink_router import (
    RouterPlan, derive_split_counts, _fused_route,
)


# ============================================================
# #1 derive_split_counts
# ============================================================
def _ground_truth_counts(per_rank, route, ep_size, num_experts):
    """Brute-force send/recv matrices from the demand matrix + routing.

    send[me][s] = Σ_{e: route[e]==s} per_rank[me][e]   (me's tokens → rank s)
    recv[me][r] = Σ_{e: route[e]==me} per_rank[r][e]    (rank r's tokens → me)
    """
    send = [[0] * ep_size for _ in range(ep_size)]
    recv = [[0] * ep_size for _ in range(ep_size)]
    for me in range(ep_size):
        for e in range(num_experts):
            r = route[e]
            if r < 0:
                continue
            send[me][r] += per_rank[me][e]
        for e in range(num_experts):
            if route[e] == me:
                for src in range(ep_size):
                    recv[me][src] += per_rank[src][e]
    return send, recv


def test_derive_split_counts_matches_ground_truth():
    torch.manual_seed(0)
    ep_size, num_experts, layer = 4, 16, 7
    # random demand matrix (per-rank per-expert token counts)
    per_rank_t = torch.randint(0, 5, (ep_size, num_experts), dtype=torch.int64)
    per_rank = per_rank_t.tolist()
    # routing: expert e -> rank (naive e % ep_size), some experts undemanded
    route = [e % ep_size for e in range(num_experts)]
    routing_map = {(layer, e): route[e] for e in range(num_experts)}
    # also stuff a different-layer entry to ensure layer filtering works
    routing_map[(layer + 1, 0)] = (route[0] + 1) % ep_size

    gt_send, gt_recv = _ground_truth_counts(per_rank, route, ep_size, num_experts)

    for me in range(ep_size):
        send, recv = derive_split_counts(
            per_rank, routing_map, layer, me, ep_size)
        assert send == gt_send[me], f"send rank{me}: {send} != {gt_send[me]}"
        assert recv == gt_recv[me], f"recv rank{me}: {recv} != {gt_recv[me]}"


def test_derive_split_counts_a2a_symmetry():
    """send[me][s] (me's view) must equal recv[s][me] (s's view) — the
    all_to_all consistency invariant.  A mismatch here = collective deadlock."""
    torch.manual_seed(1)
    ep_size, num_experts, layer = 4, 32, 3
    per_rank = torch.randint(0, 7, (ep_size, num_experts), dtype=torch.int64).tolist()
    route = [(e * 3 + 1) % ep_size for e in range(num_experts)]
    routing_map = {(layer, e): route[e] for e in range(num_experts)}

    sends, recvs = {}, {}
    for me in range(ep_size):
        s, r = derive_split_counts(per_rank, routing_map, layer, me, ep_size)
        sends[me], recvs[me] = s, r
    for a in range(ep_size):
        for b in range(ep_size):
            assert sends[a][b] == recvs[b][a], (
                f"asymmetry: send[{a}][{b}]={sends[a][b]} != "
                f"recv[{b}][{a}]={recvs[b][a]}")


def test_derive_split_counts_only_my_layer():
    """Entries for other layers must not pollute this layer's counts."""
    ep_size, num_experts, layer = 2, 4, 5
    per_rank = [[10, 0, 0, 0], [0, 0, 3, 0]]
    routing_map = {(layer, 0): 0, (layer, 2): 1, (layer + 9, 1): 0}
    send0, recv0 = derive_split_counts(per_rank, routing_map, layer, 0, ep_size)
    # rank0 sends its 10 tokens (expert0 -> rank0) to itself; expert1 ignored
    assert send0 == [10, 0]
    # rank0 receives expert0 from rank0 (10) and rank1 (0) -> [10, 0]
    assert recv0 == [10, 0]


# ============================================================
# #2 _fused_route byte round-trip
# ============================================================
def _identity_all_to_all(monkeypatch):
    """Patch dist.all_to_all_single to a same-size identity copy (single-rank
    semantics: output_split == input_split == [K])."""
    def fake(out, inp, output_split_sizes=None, input_split_sizes=None,
             group=None):
        out.copy_(inp)
    monkeypatch.setattr(nvlink_router.dist, "all_to_all_single", fake)


def _make_plan(K, H, hidden_dtype, w_dtype):
    torch.manual_seed(K + H)
    return RouterPlan(
        token_idx=torch.arange(K),
        expert_idx=torch.randint(0, 128, (K,), dtype=torch.int64),
        weights=torch.rand(K, dtype=w_dtype),
        send_hidden=torch.randn(K, H, dtype=hidden_dtype),
        send_counts=[K],
    )


def test_fused_route_roundtrip_bf16(monkeypatch):
    _identity_all_to_all(monkeypatch)
    K, H = 7, 2048   # row = 2048*2 + 8 + 4 = 4108 (NOT divisible by 8)
    plan = _make_plan(K, H, torch.bfloat16, torch.float32)
    rh, re, rw = _fused_route(plan, recv_counts=[K], K_recv=K, ep_group=None)
    assert torch.equal(rh, plan.send_hidden), "hidden corrupted"
    assert torch.equal(re, plan.expert_idx), "expert_idx corrupted"
    assert torch.equal(rw, plan.weights), "weights corrupted"
    assert rh.shape == (K, H) and re.shape == (K,) and rw.shape == (K,)


def test_fused_route_roundtrip_other_dtypes(monkeypatch):
    _identity_all_to_all(monkeypatch)
    K, H = 5, 64
    plan = _make_plan(K, H, torch.float16, torch.bfloat16)
    rh, re, rw = _fused_route(plan, recv_counts=[K], K_recv=K, ep_group=None)
    assert torch.equal(rh, plan.send_hidden)
    assert torch.equal(re, plan.expert_idx)
    assert torch.equal(rw, plan.weights)


def test_fused_route_empty(monkeypatch):
    _identity_all_to_all(monkeypatch)
    H = 2048
    plan = RouterPlan(
        token_idx=torch.empty(0, dtype=torch.int64),
        expert_idx=torch.empty(0, dtype=torch.int64),
        weights=torch.empty(0, dtype=torch.float32),
        send_hidden=torch.empty(0, H, dtype=torch.bfloat16),
        send_counts=[0],
    )
    rh, re, rw = _fused_route(plan, recv_counts=[0], K_recv=0, ep_group=None)
    assert rh.shape == (0, H) and re.shape == (0,) and rw.shape == (0,)
