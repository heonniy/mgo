"""Golden Test E — 2-rank all-to-all symmetry smoke (revision.md §3.5 / Phase 12).

No model, no archer.  Drives nvlink_router's pack_tokens / route_tokens /
return_outputs / scatter_combine_into exactly as ep_executor._coop_dispatch
does, with a unified expert_rank_table, and asserts:
  1. every rank calls route_tokens once and return_outputs once
  2. a rank with an EMPTY local plan (0 tokens) still participates (no hang)
  3. identity experts (weight 1) round-trip: final == original hidden
  4. NCCL completes (a final barrier returns)

Run:
  torchrun --standalone --nproc_per_node=2 tests/test_coslot_a2a.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from moe_infinity_ep.exec.nvlink_router import (
    build_expert_rank_table, pack_tokens, return_outputs, route_tokens,
)


def scatter_combine_into(target, send_back, token_idx, dtype):
    # mirror of ep_executor.scatter_combine_into (inlined to avoid importing
    # ep_executor, which pulls nvtx).
    if send_back.numel() == 0:
        return
    target.index_add_(0, token_idx, send_back.to(dtype))


def _one_route_return(hidden, router_mask, weights_mask, table, ep_group,
                      ep_size, num_experts, dtype, device, layer_id):
    N, H = hidden.shape
    rplan = pack_tokens(hidden, router_mask, weights_mask, table, ep_size)
    recv_hidden, recv_expert_idx, recv_weights, recv_counts = route_tokens(
        rplan, ep_group, layer_id=layer_id)
    # identity "expert" compute: return recv tokens scaled by their weight.
    local_out = recv_hidden * recv_weights.unsqueeze(1).to(recv_hidden.dtype)
    send_back = return_outputs(
        local_out, recv_counts, rplan.send_counts, ep_group, layer_id=layer_id)
    final = torch.zeros(N, H, dtype=dtype, device=device)
    scatter_combine_into(final, send_back, rplan.token_idx, dtype)
    return final, int(recv_hidden.size(0))


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, "Test E expects --nproc_per_node=2"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    num_experts, H = 4, 8
    # expert_rank_table: naive expert%ep_size (experts 0,2→r0; 1,3→r1).
    e2r = {(0, e): e % world for e in range(num_experts)}
    table = build_expert_rank_table(e2r, 0, num_experts, device)

    # ---- case 1: rank0 has 3 tokens (all expert 0), rank1 has 0 tokens ----
    if rank == 0:
        N = 3
        hidden = (torch.arange(N * H, dtype=torch.float32, device=device)
                  .reshape(N, H).to(dtype) + 1.0)
        rmask = torch.zeros(N, num_experts, dtype=torch.bool, device=device)
        rmask[:, 0] = True
        wmask = torch.zeros(N, num_experts, dtype=torch.float32, device=device)
        wmask[:, 0] = 1.0
    else:
        N = 0
        hidden = torch.zeros(0, H, dtype=dtype, device=device)
        rmask = torch.zeros(0, num_experts, dtype=torch.bool, device=device)
        wmask = torch.zeros(0, num_experts, dtype=torch.float32, device=device)

    final, k_recv = _one_route_return(
        hidden, rmask, wmask, table, ep_group, world, num_experts, dtype,
        device, layer_id=0)
    # identity weight=1 → final == original hidden (rank0); rank1 has none.
    if rank == 0:
        assert torch.allclose(final.float(), hidden.float()), \
            f"rank0 round-trip mismatch\n{final}\n{hidden}"
        # rank0 serves expert 0 → receives its own 3 tokens.
        assert k_recv == 3, f"rank0 k_recv={k_recv}"
    else:
        # rank1 serves experts 1,3 → no demand → receives 0; still participated.
        assert k_recv == 0, f"rank1 k_recv={k_recv}"

    # ---- case 2: both ranks route to each other (cross traffic) ----
    # rank r has 2 tokens selecting expert (1-r)%? → route to the other rank.
    N = 2
    hidden2 = (torch.arange(N * H, dtype=torch.float32, device=device)
               .reshape(N, H).to(dtype) + 10.0 * (rank + 1))
    rmask2 = torch.zeros(N, num_experts, dtype=torch.bool, device=device)
    other_expert = 1 if rank == 0 else 0   # expert1→r1, expert0→r0 (cross)
    rmask2[:, other_expert] = True
    wmask2 = torch.zeros(N, num_experts, dtype=torch.float32, device=device)
    wmask2[:, other_expert] = 1.0
    final2, k_recv2 = _one_route_return(
        hidden2, rmask2, wmask2, table, ep_group, world, num_experts, dtype,
        device, layer_id=1)
    assert torch.allclose(final2.float(), hidden2.float()), \
        f"rank{rank} case2 round-trip mismatch"

    dist.barrier()
    if rank == 0:
        print("Test E (2-rank a2a symmetry): ALL PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
