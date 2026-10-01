#!/usr/bin/env python3
"""Isolated NUMA-bound H2D and NCCL payload bandwidth; not model throughput."""
import argparse
import json
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT = pin_rank_before_cuda_import()

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--samples", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    rank, world = dist.get_rank(), dist.get_world_size()
    dist.all_reduce(torch.ones(1, device="cuda"))
    rows = []
    for size in (9 * 1024**2, 64 * 1024**2):
        host = torch.full((size,), 37, dtype=torch.uint8, pin_memory=True)
        slot = torch.empty(size, dtype=torch.uint8, device="cuda")
        staging = torch.empty_like(slot)
        for mode in ("direct_h2d", "staged_h2d_plus_d2d"):
            def copy():
                if mode == "direct_h2d":
                    slot.copy_(host, non_blocking=True)
                else:
                    staging.copy_(host, non_blocking=True)
                    slot.copy_(staging)
            for _ in range(args.warmup):
                copy()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.samples):
                dist.barrier()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    copy()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / args.iterations)
            assert bool((slot == 37).all())
            rows.append({"kind": mode, "host_payload_bytes": size, "milliseconds": samples,
                         "payload_GBps": [size / (ms * 1e6) for ms in samples]})
        del host, slot, staging
    size = 64 * 1024**2
    send = torch.full((size,), rank, dtype=torch.uint8, device="cuda")
    recv = torch.empty_like(send)
    for _ in range(args.warmup):
        dist.all_to_all_single(recv, send)
    samples = []
    for _ in range(args.samples):
        dist.barrier()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            dist.all_to_all_single(recv, send)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / args.iterations)
    for source, chunk in enumerate(recv.chunk(world)):
        assert bool((chunk == source).all())
    peer_bytes = size * (world - 1) // world
    rows.append({"kind": "nccl_all_to_all", "peer_payload_bytes": peer_bytes,
                 "milliseconds": samples, "payload_GBps": [peer_bytes / (ms * 1e6) for ms in samples]})
    result = {"status": "PASS", "rank": rank, "world": world, "boot": BOOT,
              "iterations_per_sample": args.iterations, "warmup_iterations": args.warmup, "measurements": rows,
              "note": "Per-rank transmitted payload; excludes wire overhead. All ranks run concurrently."}
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
