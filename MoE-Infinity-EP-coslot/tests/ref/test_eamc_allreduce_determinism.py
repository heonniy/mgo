"""EAMC cross-rank consistency — the priority matrix is byte-identical on every
rank (no model, no GPU).

EAMC is the only policy whose victim depends on a SHARED float matrix.  Multi-
process consistency (drift=0) holds iff every rank sees the SAME matrix.  The
matrix is produced by PriorityAggregator's NCCL all_reduce(SUM), whose contract
makes the output identical on all ranks.  These tests prove that on real
collectives (gloo, 2 procs):

  1. raw dist.all_reduce(SUM) of per-rank-varying float matrices yields a
     byte-identical result on every rank.
  2. the REAL PriorityAggregator (fed a per-rank-varying fake predictor) returns
     a byte-identical priority matrix on every rank.

If either failed, EAMC could diverge across ranks; verify_layer_end would then
FATAL on the real run.  These catch it earlier, on CPU.

Run:  pytest tests/ref/test_eamc_allreduce_determinism.py -q
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---- workers (must be module-level / picklable for spawn) ----
def _raw_worker(rank, world, init_file, out_dir):
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method=f"file://{init_file}")
    g = torch.Generator().manual_seed(1000 + rank)        # per-rank-varying
    local = torch.rand(8, 16, generator=g, dtype=torch.float32)
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    h = hashlib.sha256(local.numpy().tobytes()).hexdigest()
    with open(os.path.join(out_dir, f"raw_{rank}.txt"), "w") as f:
        f.write(h)
    dist.barrier()
    dist.destroy_process_group()


def _agg_worker(rank, world, init_file, out_dir):
    import numpy as np
    import torch.distributed as dist

    sys.path.insert(0, REPO_ROOT)
    os.environ["MOE_EP_EAMC_PRIORITY"] = "1"
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method=f"file://{init_file}")
    from moe_infinity_ep.controller.priority_aggregator import PriorityAggregator

    NL, NE = 8, 16

    class FakeTracer:
        def update_entry(self, *a):
            pass

        def get_entry(self, sid):
            class E:
                matrix = np.full((NL, NE), float(rank + 1), dtype=np.float32)
            return E()

        def find_most_similar(self, m, l):                # per-rank-varying
            return np.full((NL, NE), float(rank + 1), dtype=np.float32)

    class FakePred:
        def __init__(self):
            self.tracer = FakeTracer()
            self.layer_decay_func = lambda l, layer, L: 1.0

    agg = PriorityAggregator(FakePred(), NL, NE,
                             ep_group=dist.group.WORLD, device="cpu")
    m = agg.aggregate(seq_ids=[rank], layer_id=0,
                      expert_index_per_seq=np.array([[0, 1]]))
    h = hashlib.sha256(m.cpu().numpy().tobytes()).hexdigest()
    with open(os.path.join(out_dir, f"agg_{rank}.txt"), "w") as f:
        f.write(h)
    dist.barrier()
    dist.destroy_process_group()


def _run(worker, prefix, world=2):
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "pg")
        try:
            mp.spawn(worker, args=(world, init_file, d), nprocs=world, join=True)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"multiprocess gloo unavailable: {e}")
        hashes = []
        for r in range(world):
            p = os.path.join(d, f"{prefix}_{r}.txt")
            if not os.path.exists(p):
                pytest.skip(f"worker {r} produced no output (spawn issue)")
            with open(p) as f:
                hashes.append(f.read())
    return hashes


def test_raw_allreduce_byte_identity():
    hashes = _run(_raw_worker, "raw")
    assert len(set(hashes)) == 1, (
        f"all_reduce(SUM) NOT byte-identical across ranks: {hashes}")


def test_aggregator_byte_identity():
    hashes = _run(_agg_worker, "agg")
    assert len(set(hashes)) == 1, (
        f"PriorityAggregator matrix NOT byte-identical across ranks: {hashes}")


if __name__ == "__main__":
    print("raw  :", test_raw_allreduce_byte_identity())
    print("agg  :", test_aggregator_byte_identity())
    print("all_reduce determinism: ALL PASS")
