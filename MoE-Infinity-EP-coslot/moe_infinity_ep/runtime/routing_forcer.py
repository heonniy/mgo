"""Routing record/force for fair cross-policy ablation.

Why: cache HIT and tpot both depend on the per-token expert *selection*
(demand).  On GPU the selection drifts across placement policies (cuBLAS +
placement-dependent a2a FP noise -> top-k flips), so a free-run comparison is
unfair.  This records the routing (top-k expert ids + weights) from one
reference run and *forces* the identical routing into every other policy's run,
so demand is byte-identical -> hit / a2a / fetch / tpot all measured on the same
input.  OFF by default (free-run unaffected).

Modes (env):
  MOE_EP_ROUTING_RECORD_DIR=<dir>  -> record this run's routing to <dir>.
  MOE_EP_ROUTING_FORCE_DIR=<dir>   -> override routing from <dir>.
  (neither set -> "off", normal free-run, zero hot-path cost.)

Call sequence (run_layer order) is identical across policies because the bench
streams the same batches/steps/layers with teacher-forced tokens, so a per-rank
monotonic counter aligns record<->force.
"""
from __future__ import annotations

import atexit
import os
import torch


class RoutingForcer:
    def __init__(self, rank: int) -> None:
        self.rank = int(rank)
        rec = os.environ.get("MOE_EP_ROUTING_RECORD_DIR", "")
        frc = os.environ.get("MOE_EP_ROUTING_FORCE_DIR", "")
        self.mode = "record" if rec else ("force" if frc else "off")
        self._dir = rec or frc
        self._buf: list = []      # record: [(idx[T,k] int16, w[T,k] fp16), ...]
        self._forced = None       # force: same list, replayed by counter
        self._ctr = 0
        if self.mode == "force":
            path = os.path.join(self._dir, f"routing_rank{self.rank}.pt")
            self._forced = torch.load(path, map_location="cpu")
            print(f"[routing_forcer] FORCE rank{self.rank}: "
                  f"{len(self._forced)} calls from {path}", flush=True)
        elif self.mode == "record":
            atexit.register(self.save)
            print(f"[routing_forcer] RECORD rank{self.rank} -> {self._dir}",
                  flush=True)

    def apply(self, router_mask: torch.Tensor,
              routing_weights_mask: torch.Tensor):
        """router_mask/routing_weights_mask: [T, E].  Returns possibly-overridden
        pair.  router_mask is multi-hot (exactly top_k True per row)."""
        if self.mode == "off":
            return router_mask, routing_weights_mask
        E = router_mask.shape[1]
        if self.mode == "record":
            b = router_mask.bool()
            T = b.shape[0]
            # sparse (row,col) pairs — handles variable/zero experts-per-row
            # (e.g. padding tokens), no fixed-k reshape assumption.
            rc = b.nonzero(as_tuple=False)                 # [nnz, 2] (row, col)
            w = routing_weights_mask[rc[:, 0], rc[:, 1]].to(torch.float16)
            self._buf.append((rc.to(torch.int32).cpu(), w.cpu(), int(T)))
            return router_mask, routing_weights_mask
        # force
        rc, w, T = self._forced[self._ctr]
        self._ctr += 1
        dev = router_mask.device
        rc = rc.to(dev).long(); w = w.to(dev)
        rm = torch.zeros(T, E, dtype=router_mask.dtype, device=dev)
        rm[rc[:, 0], rc[:, 1]] = 1
        rw = torch.zeros(T, E, dtype=routing_weights_mask.dtype, device=dev)
        rw[rc[:, 0], rc[:, 1]] = w.to(routing_weights_mask.dtype)
        return rm, rw

    def save(self) -> None:
        if self.mode != "record" or not self._buf:
            return
        os.makedirs(self._dir, exist_ok=True)
        path = os.path.join(self._dir, f"routing_rank{self.rank}.pt")
        torch.save(self._buf, path)
        print(f"[routing_forcer] SAVED rank{self.rank}: {len(self._buf)} "
              f"calls -> {path}", flush=True)
        self._buf = []   # avoid double-save (atexit + explicit)
