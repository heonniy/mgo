"""EAMC live-wiring (Stage 1) — eviction-only hook, no prefetch, no GPU.

Verifies the controller<-aggregator injection path added for real EAMC:
  * maybe_inject_eamc_priority is a NO-OP without an aggregator (so normal
    LRU/LFU runs are byte-identical — zero overhead).
  * with an aggregator attached it sets the layer priority, and plan_misses
    then evicts the min-priority victim.
  * EVICTION-ONLY: the hook never adds a fetch — #fetch_ops == #misses always
    (no future-layer prefetch is introduced).
  * best-effort: an aggregator that raises -> priority None -> LRU fallback,
    never crashes the layer.
  * a real PriorityAggregator (with a fake predictor) returns a matrix.

Run:  pytest tests/ref/test_eamc_wiring.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from reference import demand_matrix          # noqa: E402
import prod_driver as pd                       # noqa: E402

L = 0
NL = 64


def make_priority(ne, row0, num_layers=NL):
    m = [[1.0] * ne for _ in range(num_layers)]
    for e, s in row0.items():
        m[0][e] = float(s)
    return m


class FakeAgg:
    """Stand-in PriorityAggregator: returns a fixed matrix, counts calls."""

    def __init__(self, matrix):
        self.matrix = matrix
        self.calls = 0

    def aggregate(self, layer_id, **kwargs):
        self.calls += 1
        return self.matrix


# ------------------------------------------------------------------
# 1. no aggregator -> hook is a pure no-op (normal runs unaffected).
# ------------------------------------------------------------------
def test_hook_noop_without_aggregator():
    c = pd.make_prod(2, 2, 8, "naive", "eamc")
    assert c.priority_aggregator is None
    assert c._layer_priority is None
    c.maybe_inject_eamc_priority(0)            # must do nothing
    assert c._layer_priority is None


# ------------------------------------------------------------------
# 2. aggregator attached -> priority injected -> min-priority eviction.
#    Also: EVICTION-ONLY (#fetch_ops == #misses, no prefetch).
# ------------------------------------------------------------------
def test_hook_injects_and_evicts_min_priority():
    pr = make_priority(8, {0: 0.9, 1: 0.1})    # E1 lowest -> victim
    c = pd.make_prod(2, 2, 8, "naive", "eamc")
    pd.seed_prod(c, {0: [(L, 0), (L, 1)], 1: []})
    c.priority_aggregator = FakeAgg(pr)

    c.maybe_inject_eamc_priority(0)            # ep_executor hook
    assert c.priority_aggregator.calls == 1
    assert c._layer_priority is pr

    a = pd.prod_step(c, demand_matrix(2, 8, {0: {2: 5}}), L)  # E2 miss on rank0
    fos = a["fetch_ops"]
    assert len(fos) == 1 and fos[0]["victim_expert"] == (L, 1)
    # EVICTION-ONLY: no extra fetch beyond the actual miss
    assert len(fos) == len(a["misses"]), "hook must NOT add prefetch fetches"


# ------------------------------------------------------------------
# 3. best-effort: aggregator that raises -> None -> LRU fallback.
# ------------------------------------------------------------------
def test_hook_best_effort_on_aggregator_failure():
    class BadAgg:
        def aggregate(self, layer_id, **kwargs):
            raise RuntimeError("predict blew up")

    c = pd.make_prod(2, 2, 8, "naive", "eamc")
    c.set_layer_priority("stale-matrix")       # something non-None
    c.priority_aggregator = BadAgg()
    c.maybe_inject_eamc_priority(0)            # swallow -> None
    assert c._layer_priority is None


# ------------------------------------------------------------------
# 4. a REAL PriorityAggregator with a fake predictor returns a matrix
#    (single process: no dist -> no all_reduce, just the local matrix).
# ------------------------------------------------------------------
def test_real_aggregator_with_fake_predictor():
    from moe_infinity_ep.controller.priority_aggregator import PriorityAggregator

    NE = 8

    class FakeTracer:
        def update_entry(self, seq_id, arr, layer_id):
            pass

        def get_entry(self, seq_id):
            class E:
                matrix = np.ones((NL, NE), dtype=np.float32)
            return E()

        def find_most_similar(self, matrix, layer_id):
            return np.ones((NL, NE), dtype=np.float32)

    class FakePredictor:
        def __init__(self):
            self.tracer = FakeTracer()
            self.layer_decay_func = lambda l, layer, L: 1.0

    prev = os.environ.get("MOE_EP_EAMC_PRIORITY")
    os.environ["MOE_EP_EAMC_PRIORITY"] = "1"
    try:
        agg = PriorityAggregator(FakePredictor(), num_layers=NL, num_experts=NE,
                                 ep_group=None, device="cpu")
        assert agg.enabled()
        m = agg.aggregate(seq_ids=[0], layer_id=0,
                          expert_index_per_seq=np.array([[0, 1]]))
        assert m is not None and tuple(m.shape) == (NL, NE)
        # off-stride layer (layer 1, stride 8) -> None -> LRU fallback
        assert agg.aggregate(seq_ids=[0], layer_id=1,
                             expert_index_per_seq=np.array([[0, 1]])) is None
    finally:
        if prev is None:
            os.environ.pop("MOE_EP_EAMC_PRIORITY", None)
        else:
            os.environ["MOE_EP_EAMC_PRIORITY"] = prev


def test_router_mask_to_expert_index():
    import torch

    from moe_infinity_ep.controller.global_controller import GlobalCacheController as GC

    # b == n: per-row selected experts
    rm = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.float32)
    out = GC._router_mask_to_expert_index(rm, seq_ids=["a", "b"])
    assert [sorted(x.tolist()) for x in out] == [[0, 2], [1]]
    # b == 1: all rows fold into the single seq
    out1 = GC._router_mask_to_expert_index(rm, seq_ids=["s"])
    assert sorted(out1[0].tolist()) == [0, 1, 2]
    # n % b == 0: even row grouping (e.g. prefill S tokens / seq)
    rm4 = torch.eye(4, dtype=torch.float32)
    out2 = GC._router_mask_to_expert_index(rm4, seq_ids=["a", "b"])
    assert sorted(out2[0].tolist()) == [0, 1] and sorted(out2[1].tolist()) == [2, 3]
    # degenerate / incompatible -> None (best-effort, never raises)
    assert GC._router_mask_to_expert_index(None, ["a"]) is None
    assert GC._router_mask_to_expert_index(rm, None) is None
    rm3 = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
    assert GC._router_mask_to_expert_index(rm3, ["a", "b"]) is None   # 3 % 2 != 0


if __name__ == "__main__":
    test_hook_noop_without_aggregator()
    test_hook_injects_and_evicts_min_priority()
    test_hook_best_effort_on_aggregator_failure()
    test_real_aggregator_with_fake_predictor()
    print("EAMC wiring: ALL PASS")
