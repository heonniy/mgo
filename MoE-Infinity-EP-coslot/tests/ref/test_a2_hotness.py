"""Stage-2 A2 live estimator tests (a2_hotness.py).

  * A2Store.query_topR overlap counts vs an independent brute-force cosine/overlap
    golden (A2_algorithm.md: full-W cosine == shared-cell count).
  * A2Store.predict vs brute-force top-R future averaging (tie-free / R>=pos cases).
  * TrajectoryRecorder round-trip.
  * A2HotnessProvider aggregation (A2Hotness/A2Demand), all_reduce hook, cold-start.

Run:  pytest tests/ref/test_a2_hotness.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from moe_infinity_ep.controller.a2_hotness import (   # noqa: E402
    A2Meta, A2Store, RequestData, TrajectoryRecorder, A2HotnessProvider,
    GlobalFreqTracker, EvictCostProvider, hist_window, normalize_rows)


def _rand_req(rng, meta, T, ordinal):
    sel = np.stack([
        np.stack([rng.choice(meta.E, size=meta.K, replace=False)
                  for _ in range(meta.L)], axis=0)
        for _ in range(T)], axis=0)            # [T, L, K]
    return RequestData(T=T, sel=sel.astype(np.int64), ord=ordinal)


def _cells(sel, lo, hi, base):
    """Set of (pos, l, e) cells for steps [lo, hi); pos = step - base."""
    L, K = sel.shape[1], sel.shape[2]
    s = set()
    for step in range(lo, hi):
        pos = step - base
        for l in range(L):
            for e in sel[step, l, :]:
                s.add((pos, l, int(e)))
    return s


def _brute_overlaps(store, reqs, qr, t, cutoff_ord):
    """Independent overlap counts for every eligible (r', u') sample."""
    W, H = store.W, store.H
    qcells = _cells(qr.sel, max(0, t - W), t, t - W)
    out = []
    for r in reqs:
        if r.ord >= cutoff_ord:
            continue
        for u in range(W, r.T - H + 1):
            sc = _cells(r.sel, u - W, u, u - W)
            out.append((len(qcells & sc), r, u))
    return out


def test_query_topR_counts_vs_brute():
    rng = np.random.default_rng(0)
    meta = A2Meta(L=3, E=8, K=2)
    W = H = 4
    for trial in range(30):
        reqs = [_rand_req(rng, meta, T=int(rng.integers(W + H + 1, W + H + 6)),
                          ordinal=i) for i in range(int(rng.integers(2, 6)))]
        store = A2Store(meta, W=W, H=H)
        for r in reqs:
            store.add_request(r)
        store.finalize()
        qr = reqs[-1]
        t = int(rng.integers(1, qr.T))
        cutoff = qr.ord
        Rmax = int(rng.integers(1, 8))
        top, cnts = store.query_topR(qr, t, cutoff, Rmax=Rmax)
        brute = sorted((c for c, _, _ in _brute_overlaps(store, reqs, qr, t, cutoff)
                        if c > 0), reverse=True)[:Rmax]
        assert cnts == brute, (f"trial={trial} cnts={cnts} brute={brute}")
        # determinism
        assert store.query_topR(qr, t, cutoff, Rmax=Rmax)[1] == cnts


def test_predict_vs_brute_all_positive():
    """When R >= #positive-overlap samples, predict == normalized sum of all
    positive samples' futures (no tie ambiguity)."""
    rng = np.random.default_rng(7)
    meta = A2Meta(L=2, E=6, K=2)
    W = H = 3
    for trial in range(30):
        reqs = [_rand_req(rng, meta, T=int(rng.integers(W + H + 1, W + H + 5)),
                          ordinal=i) for i in range(int(rng.integers(2, 5)))]
        store = A2Store(meta, W=W, H=H)
        for r in reqs:
            store.add_request(r)
        store.finalize()
        qr = reqs[-1]
        t = int(rng.integers(1, qr.T))
        ov = [(c, r, u) for (c, r, u) in _brute_overlaps(store, reqs, qr, t, qr.ord)
              if c > 0]
        npos = len(ov)
        if npos == 0:
            assert store.predict(qr, t, qr.ord, R=4) is None
            continue
        R = npos + 2            # cover all positive -> no boundary tie
        pred = store.predict(qr, t, qr.ord, R=R)
        acc = np.zeros((meta.L, meta.E))
        for (_c, r, u) in ov:
            acc += normalize_rows(hist_window(r.sel, u, H, meta.L, meta.E))
        assert pred is not None
        np.testing.assert_allclose(pred, normalize_rows(acc), atol=1e-9)


def test_cold_start_empty_store():
    meta = A2Meta(L=2, E=4, K=1)
    store = A2Store(meta, W=2, H=2).finalize()       # no requests
    qr = RequestData(T=3, sel=np.zeros((3, 2, 1), np.int64), ord=5)
    assert store.query_topR(qr, 2, 5) == ([], [])
    assert store.predict(qr, 2, 5, R=4) is None


def test_recorder_roundtrip():
    meta = A2Meta(L=3, E=8, K=2)
    rec = TrajectoryRecorder(meta)
    sid = "seqA"
    expected = []
    for step in range(4):
        rec.start_step(sid)
        step_sel = []
        for l in range(meta.L):
            ids = [(step + l) % meta.E, (step + l + 1) % meta.E]
            rec.record(sid, l, ids)
            step_sel.append(ids)
        expected.append(step_sel)
    r = rec.finish(sid, ordinal=42)
    assert r is not None and r.T == 4 and r.ord == 42
    assert r.sel.shape == (4, meta.L, meta.K)
    np.testing.assert_array_equal(r.sel, np.array(expected, dtype=np.int64))
    # buffer freed
    assert rec.finish(sid, 0) is None


def test_recorder_short_topk_padding():
    meta = A2Meta(L=1, E=8, K=4)
    rec = TrajectoryRecorder(meta)
    rec.start_step("s")
    rec.record("s", 0, [3])               # fewer than K -> padded
    r = rec.finish("s", 1)
    assert r.sel.shape == (1, 1, 4)
    assert r.sel[0, 0, 0] == 3 and (r.sel[0, 0] == 3).all()


def test_provider_aggregation_and_reduce():
    meta = A2Meta(L=2, E=3, K=1)
    prov = A2HotnessProvider(meta, ep_size=2)
    pA = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])   # request on rank 0
    pB = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])   # request on rank 1
    prov.set_step([(pA, 0), (pB, 1)])
    # A2Demand per rank
    assert prov.future_affinity((0, 0), 0) == 1.0
    assert prov.future_affinity((0, 0), 1) == 0.0
    assert prov.future_affinity((0, 2), 1) == 3.0
    assert prov.future_affinity((1, 1), 0) == 2.0
    # A2Hotness = sum over ranks
    assert prov.evict_cost((0, 0)) == 1.0
    assert prov.evict_cost((0, 2)) == 3.0
    assert prov.evict_cost((1, 1)) == 2.0
    # out-of-range safe
    assert prov.evict_cost((99, 0)) == 0.0
    assert prov.future_affinity((0, 0), 99) == 0.0
    # all_reduce hook doubles (simulating a 2-rank sum of identical locals)
    prov.set_step([(pA, 0)], all_reduce_fn=lambda x: x * 2.0)
    assert prov.future_affinity((0, 0), 0) == 2.0
    # clear
    prov.clear()
    assert prov.evict_cost((0, 0)) == 0.0


def test_provider_none_preds():
    meta = A2Meta(L=2, E=3, K=1)
    prov = A2HotnessProvider(meta, ep_size=2)
    prov.set_step([(None, 0), (None, 1)])   # all cold-start
    assert prov.evict_cost((0, 0)) == 0.0
    assert prov.future_affinity((0, 0), 0) == 0.0


def test_global_freq_tracker_bounded_and_norm():
    meta = A2Meta(L=4, E=4, K=2)
    ft = GlobalFreqTracker(meta, capacity=2)
    r1 = np.zeros((4, 4)); r1[2, 2] = 100; r1[2, 3] = 200
    r2 = np.zeros((4, 4)); r2[2, 2] = 100; r2[2, 3] = 250
    ft.add_request(r1); ft.add_request(r2)
    assert ft.freq[2, 2] == 200 and ft.freq[2, 3] == 450     # sum of 2 reqs
    # norm per layer: L2 max = 450 → E2=200/450, E3=450/450
    nf = ft.norm_freq()
    np.testing.assert_allclose(nf[2, 2], 200 / 450, atol=1e-9)
    np.testing.assert_allclose(nf[2, 3], 1.0, atol=1e-9)
    # bounded: 3rd request drops r1 (capacity=2)
    r3 = np.zeros((4, 4)); r3[2, 2] = 10
    ft.add_request(r3)
    assert ft.freq[2, 2] == 100 + 10 and ft.freq[2, 3] == 250  # r1 evicted


def test_evict_cost_worked_example():
    """The (L2,E2) vs (L3,E3) worked example from EVICTION_COST_TERMS.md:
       NormFreq(L2,E2)=200/450=.444, Future=.60 → cost=.60*(1+.5*.444)=0.7333
       NormFreq(L3,E3)=120/600=.20,  Future=.15 → cost=.15*(1+.5*.20)=0.165
    """
    meta = A2Meta(L=4, E=4, K=2)
    ft = GlobalFreqTracker(meta, capacity=1000)
    counts = np.zeros((4, 4))
    counts[2] = [300, 50, 200, 450]      # L2 freqs
    counts[3] = [100, 600, 80, 120]      # L3 freqs
    ft.add_request(counts)
    prov = EvictCostProvider(meta, mu=0.5, freq_tracker=ft)
    future = np.zeros((4, 4))
    future[2] = [0.1, 0.0, 0.60, 0.3]    # predicted future (already layer-decayed)
    future[3] = [0.0, 0.7, 0.10, 0.15]
    prov.set_future(future)
    np.testing.assert_allclose(prov.evict_cost((2, 2)), 0.60 * (1 + 0.5 * (200 / 450)), atol=1e-9)
    np.testing.assert_allclose(prov.evict_cost((3, 3)), 0.15 * (1 + 0.5 * (120 / 600)), atol=1e-9)
    # (L2,E2) much riskier than (L3,E3) → keep E2@L2, evict E3@L3
    assert prov.evict_cost((2, 2)) > prov.evict_cost((3, 3))


def test_evict_cost_cold_start_fallback():
    meta = A2Meta(L=2, E=3, K=1)
    ft = GlobalFreqTracker(meta, capacity=10)
    c = np.zeros((2, 3)); c[0] = [4, 2, 0]; ft.add_request(c)
    prov = EvictCostProvider(meta, mu=0.5, freq_tracker=ft)
    prov.set_future(None)                       # no future → freq-only fallback
    assert prov.evict_cost((0, 0)) == 1.0       # 4/4 normfreq
    np.testing.assert_allclose(prov.evict_cost((0, 1)), 0.5)  # 2/4
    assert prov.evict_cost((0, 2)) == 0.0


def test_evict_cost_product_vs_correction_note():
    """Sanity: with freq=0 the correction form keeps the future signal (×1),
    unlike a pure product which would zero it out."""
    meta = A2Meta(L=1, E=2, K=1)
    ft = GlobalFreqTracker(meta, capacity=10)   # empty → normfreq all 0
    prov = EvictCostProvider(meta, mu=0.5, freq_tracker=ft)
    prov.set_future(np.array([[0.0, 0.8]]))
    assert prov.evict_cost((0, 1)) == 0.8       # future survives (×(1+0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
