"""Sparse active-peer NCCL send/recv transport."""
from __future__ import annotations
import torch
import torch.distributed as dist

def split_offsets(counts):
    out=[0]
    for x in counts:out.append(out[-1]+int(x))
    return out

def active_peer_set(send_counts,recv_counts,rank):
    return tuple(peer for peer,(s,r) in enumerate(zip(send_counts,recv_counts)) if peer!=rank and (int(s)>0 or int(r)>0))

def exchange_active_peers(send:torch.Tensor,send_counts,recv_counts):
    rank=dist.get_rank();world=dist.get_world_size()
    if len(send_counts)!=world or len(recv_counts)!=world:raise ValueError("count length != world")
    if sum(int(x) for x in send_counts)!=send.shape[0]:raise ValueError("send count mismatch")
    recv=torch.empty((sum(int(x) for x in recv_counts),)+tuple(send.shape[1:]),dtype=send.dtype,device=send.device)
    send=send.contiguous();so=split_offsets(send_counts);ro=split_offsets(recv_counts)
    if int(send_counts[rank])!=int(recv_counts[rank]):raise RuntimeError("self count mismatch")
    if int(send_counts[rank]):recv[ro[rank]:ro[rank+1]].copy_(send[so[rank]:so[rank+1]])
    peers=active_peer_set(send_counts,recv_counts,rank);ops=[]
    for peer in peers:
        if int(recv_counts[peer]):ops.append(dist.P2POp(dist.irecv,recv[ro[peer]:ro[peer+1]],peer))
        if int(send_counts[peer]):ops.append(dist.P2POp(dist.isend,send[so[peer]:so[peer+1]],peer))
    if ops:
        for work in dist.batch_isend_irecv(ops):work.wait()
    return recv,peers,len(ops)
