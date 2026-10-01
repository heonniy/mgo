"""CommandDispatcher — translate a controller LayerPlan into ONE archer
``submit_plan`` call (Controller-Owned Slot Execution, 2026-05-29).

revision.md Phase 9.  The dispatcher no longer issues per-op evict/fetch
commands.  It filters the global plan to THIS rank's hit/miss ops and hands
the whole FetchPlan to archer's ``submit_plan(gpu, hit_ops, miss_ops)``; archer
runs the PlanQueue (one active fetch, direct or staging) and the exec workers.

Token-0 note: a planned hit/miss expert is, by construction, demanded
(demand>0) and routed to its owner/fetcher, so that rank ALWAYS receives >0
tokens for it after the forward all-to-all.  We therefore do not drop any of
this rank's ops (dropping mid-chain would break the victim chain / order
contiguity).  If a token-0 op ever reached archer it would FATAL in
ComputeSnapshot (fail-fast #8) — surfacing a controller bug rather than hiding
it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .layer_plan import LayerPlan, HitOp


_HEAVY = os.environ.get("MOE_EP_HEAVY_DEBUG", "0") == "1"


class CommandDispatcher:
    """archer ``submit_plan`` bridge for the coslot flow.

    Held by GlobalCacheController; ``local_dispatcher`` (= archer's
    ``ExpertDispatcher``) wired by ``DistributedOffloadEngine.__exit__`` once
    the model is built.
    """

    def __init__(self, archer_engine: Any, local_dispatcher: Any, rank: int):
        self.archer_engine = archer_engine
        self.local_dispatcher = local_dispatcher
        self.rank = rank

        # Stats (kept for trace.json compatibility; failures are now FATAL in
        # archer, so the *_failures counters stay 0 under correct operation).
        self.n_fetches: int = 0
        self.n_evicts: int = 0
        self.n_fetch_failures: int = 0
        self.n_evict_failures: int = 0
        self.n_evict_noops: int = 0
        self.fetch_us_list: List[int] = []
        self.evict_us_list: List[int] = []

        self._has_submit_plan = (
            self.local_dispatcher is not None
            and hasattr(self.local_dispatcher, "submit_plan")
        )
        if self.local_dispatcher is not None and not self._has_submit_plan:
            raise RuntimeError(
                "[dispatcher] archer build lacks submit_plan — rebuild the "
                "coslot archer (CUTLASS_DIR=... python setup.py build_ext "
                "--inplace).")
        print(f"[dispatcher rank={rank}] init: coslot submit_plan "
              f"available={self._has_submit_plan}", flush=True)

    # ------------------------------------------------------------
    # Phase 9 — submit this rank's slice of the plan to archer.
    # ------------------------------------------------------------
    def submit_plan(
        self,
        plan: "LayerPlan",
        hit_ops: List["HitOp"],
        my_rank: int,
    ) -> None:
        if self.local_dispatcher is None:
            return

        hit_tuples: List[Tuple[int, int, int]] = []
        for h in hit_ops:
            if h.owner_rank != my_rank:
                continue
            l, e = h.expert
            hit_tuples.append((int(l), int(e), int(h.slot)))

        miss_tuples: List[Tuple[int, int, int, int, int, int]] = []
        for op in plan.fetch_ops:
            if op.fetcher_rank != my_rank:
                continue
            l, e = op.expert
            if op.victim_expert is None:
                vl, ve = -1, -1
            else:
                vl, ve = int(op.victim_expert[0]), int(op.victim_expert[1])
            miss_tuples.append(
                (int(l), int(e), int(op.dst_slot), vl, ve, int(op.order)))

        # rank-local order MUST be a contiguous 0..n-1 prefix (PlanQueue
        # FIFO == archer expected_order_).  Fail fast here with a clear message
        # rather than triggering archer's order FATAL.
        orders = sorted(t[5] for t in miss_tuples)
        if orders != list(range(len(orders))):
            raise RuntimeError(
                f"[dispatcher rank={my_rank}] L{plan.layer_id} non-contiguous "
                f"miss order {orders} — controller bug.")

        self.local_dispatcher.submit_plan(0, hit_tuples, miss_tuples)

        self.n_fetches += len(miss_tuples)
        self.n_evicts += sum(1 for t in miss_tuples if t[3] >= 0)
        if _HEAVY:
            print(f"[dispatch rank={my_rank}] L{plan.layer_id} submit "
                  f"hits={len(hit_tuples)} miss={len(miss_tuples)}", flush=True)

    # ------------------------------------------------------------
    # bookkeeping helpers (kept for counters compatibility)
    # ------------------------------------------------------------
    def drain_latency(self) -> Tuple[List[int], List[int]]:
        f, e = self.fetch_us_list, self.evict_us_list
        self.fetch_us_list = []
        self.evict_us_list = []
        return f, e

    def snapshot(self) -> Dict[str, int]:
        return {
            "n_fetches":        self.n_fetches,
            "n_evicts":         self.n_evicts,
            "n_fetch_failures": self.n_fetch_failures,
            "n_evict_failures": self.n_evict_failures,
            "n_evict_noops":    self.n_evict_noops,
        }
