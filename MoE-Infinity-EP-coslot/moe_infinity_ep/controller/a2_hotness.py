"""Live A2 future-hotness estimator for OURS decode placement (Stage 2).

Ports the A2-FineMoE-Traj estimator from the trace_estimater experiment
(``eval_lib/traj.py``, spec: A2_algorithm.md) into a live decode loop and exposes
the two providers ``OursPlacementPlanner`` consumes:

    evict_cost_provider((l, e))        -> A2Hotness[l, e]   (DP slot cost; a hot
                                          resident is expensive to evict)
    future_affinity_provider((l, e), g)-> A2Demand[l, e, g] (decode placement
                                          future-reuse locality, per rank)

A2 idea (A2_algorithm.md): at decode step t, take the request's recent W-step
routing trajectory, find the top-R most-similar windows among PAST requests
(cosine == overlap count for full-W keys), and average their actual future-H
expert distributions -> a per-(layer, expert) future-hotness prediction [L, E].

Components here are the TESTABLE core (no runtime/torch coupling):
  * A2Meta / RequestData            — shapes
  * hist_window / normalize_rows    — inlined from candidates.py
  * A2Store                         — incremental historical store (port of
                                      A2TrajStore): add_request/finalize/
                                      query_topR/value/predict
  * TrajectoryRecorder              — per-seq per-step top-k accumulation ->
                                      RequestData on completion
  * A2HotnessProvider               — turn active requests' [L,E] predictions +
                                      a request->rank map into replicated
                                      A2Hotness[L,E] / A2Demand[L,E,G] and the
                                      two provider callables.

Cross-rank note: for the OURS plan to stay byte-identical across EP ranks, the
A2Hotness / A2Demand matrices MUST be identical on every rank.  Each rank
contributes its LOCAL requests; the caller passes an ``all_reduce_fn`` that sums
the per-rank contributions (e.g. NCCL all_reduce) before the providers are read.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


# ===================================================================
# shapes
# ===================================================================
@dataclass
class A2Meta:
    L: int   # num MoE layers
    E: int   # experts per layer
    K: int   # top-k per layer per token


@dataclass
class RequestData:
    """One request's decode routing trajectory.

    T   : number of decode steps recorded
    sel : [T, L, K] int array of selected expert ids per step/layer
    ord : request ordinal (monotonic; used for leakage filtering)
    """
    T: int
    sel: np.ndarray
    ord: int


# ===================================================================
# inlined helpers (candidates.py)
# ===================================================================
def hist_window(sel: np.ndarray, t: int, H: int, L: int, E: int) -> np.ndarray:
    """Counts [L, E] of expert activations over decode steps [t, t+H)."""
    window = sel[t:t + H]                                   # [h, L, K]
    layer_idx = np.broadcast_to(np.arange(L)[None, :, None], window.shape)
    flat = (layer_idx * E + window).ravel()
    return np.bincount(flat, minlength=L * E).reshape(L, E).astype(np.float64)


def normalize_rows(M: np.ndarray) -> np.ndarray:
    """Row-normalize [L, E] -> probabilities; zero-sum rows stay zero."""
    s = M.sum(axis=1, keepdims=True)
    out = np.zeros_like(M, dtype=np.float64)
    nz = s[:, 0] > 0
    out[nz] = M[nz] / s[nz]
    return out


# ===================================================================
# A2 store — port of eval_lib/traj.py A2TrajStore (numerically identical)
# ===================================================================
class A2Store:
    """Incremental historical trajectory store + cosine top-R retrieval.

    Direct port of ``A2TrajStore`` (trace_estimater) so it matches the validated
    reference bit-for-bit (golden-tested).  Build by ``add_request`` (past,
    completed requests only), ``finalize``, then ``query_topR`` / ``value`` /
    ``predict`` at decode time.
    """

    def __init__(self, meta: A2Meta, W: int = 16, H: int = 16):
        self.meta = meta
        self.W, self.H = W, H
        self.LE = meta.L * meta.E
        self.ncols = W * self.LE
        self._rows: List[np.ndarray] = []
        self._cols: List[np.ndarray] = []
        self.src_ord: List = []
        self.req_idx: List = []
        self.u_arr: List = []
        self.reqs: List[RequestData] = []
        self._n = 0
        self._lay = np.arange(meta.L) * meta.E
        self.csc = None

    def _stepcols(self, sel_req: np.ndarray) -> np.ndarray:
        T = sel_req.shape[0]
        return (self._lay[None, :, None] + sel_req).reshape(T, -1)

    def add_request(self, r: RequestData) -> None:
        ridx = len(self.reqs)
        self.reqs.append(r)
        W, H, LE = self.W, self.H, self.LE
        T = r.T
        ns = T - H - W + 1            # valid sample positions u in [W, T-H]
        if ns <= 0:
            return
        sc = self._stepcols(r.sel)    # [T, L*K]
        LK = sc.shape[1]
        base = self._n
        for p in range(W):
            steps = np.arange(p, p + ns)
            cols_p = sc[steps] + p * LE
            self._cols.append(cols_p.ravel().astype(np.int32))
            self._rows.append(np.repeat(
                np.arange(base, base + ns, dtype=np.int32), LK))
        self.src_ord.append(np.full(ns, r.ord, np.int64))
        self.req_idx.append(np.full(ns, ridx, np.int32))
        self.u_arr.append(np.arange(W, T - H + 1, dtype=np.int32))
        self._n += ns

    def finalize(self) -> "A2Store":
        if self._n == 0:
            self.csc = None
            return self
        rows = np.concatenate(self._rows)
        cols = np.concatenate(self._cols)
        data = np.full(rows.shape[0], 1.0 / self.meta.K, np.float32)
        self.csc = sparse.coo_matrix(
            (data, (rows, cols)), shape=(self._n, self.ncols)).tocsc()
        self.src_ord = np.concatenate(self.src_ord)
        self.req_idx = np.concatenate(self.req_idx)
        self.u_arr = np.concatenate(self.u_arr)
        self._acc = np.zeros(self._n, np.int32)
        return self

    def _query_cols(self, r: RequestData, t: int) -> Tuple[np.ndarray, int]:
        W, LE = self.W, self.LE
        lo = max(0, t - W)
        steps = np.arange(lo, t)
        pos = steps - (t - W)
        sc = (self._lay[None, :, None] + r.sel[steps]).reshape(len(steps), -1)
        cols = (sc + (pos * LE)[:, None]).ravel()
        return cols.astype(np.int64), len(steps)

    def query_topR(self, r: RequestData, t: int, cutoff_ord: int,
                   Rmax: int = 8) -> Tuple[List[int], List[int]]:
        if self.csc is None:
            return [], []
        n_elig = int(np.searchsorted(self.src_ord, cutoff_ord, side="left"))
        if n_elig == 0:
            return [], []
        cols, npos = self._query_cols(r, t)
        if npos == 0:
            return [], []
        acc = self._acc
        acc[:n_elig] = 0
        indptr, indices = self.csc.indptr, self.csc.indices
        for c in cols:
            rr = indices[indptr[c]:indptr[c + 1]]
            k = np.searchsorted(rr, n_elig, side="left")
            if k:
                acc[rr[:k]] += 1
        elig = acc[:n_elig]
        R = min(Rmax, n_elig)
        top = np.argpartition(elig, n_elig - R)[n_elig - R:]
        top = top[np.argsort(elig[top])[::-1]]
        top = [int(i) for i in top if elig[i] > 0]
        cnts = [int(elig[i]) for i in top]
        return top, cnts

    def value(self, sample_idx: int, H: int) -> np.ndarray:
        ridx = self.req_idx[sample_idx]
        u = int(self.u_arr[sample_idx])
        r = self.reqs[ridx]
        return normalize_rows(hist_window(r.sel, u, H, self.meta.L, self.meta.E))

    def predict(self, r: RequestData, t: int, cutoff_ord: int,
                R: int = 4) -> Optional[np.ndarray]:
        """Top-R averaged future-hotness [L, E] (normalized), or None if no match.

        Mirrors run_eval_h16's top-R uniform accumulation.
        """
        top, _ = self.query_topR(r, t, cutoff_ord, Rmax=R)
        if not top:
            return None
        acc = np.zeros((self.meta.L, self.meta.E), dtype=np.float64)
        for i in top[:R]:
            acc += self.value(i, self.H)
        return normalize_rows(acc)


# ===================================================================
# Per-seq trajectory recorder
# ===================================================================
class TrajectoryRecorder:
    """Accumulate per-seq, per-step, per-layer top-k expert ids during decode.

    Usage per decode step t (layers 0..L-1):
        rec.record(seq_id, layer_id, expert_ids)   # expert_ids: array[K]
    The recorder appends a new [L, K] step the first time a (seq, step) layer is
    seen; ``layer_id`` ordering within a step is arbitrary (filled by index).
    On completion: ``finish(seq_id, ord) -> RequestData`` (and frees the buffer).
    """

    def __init__(self, meta: A2Meta):
        self.meta = meta
        # seq_id -> list of [L, K] int arrays (one per recorded decode step)
        self._buf: Dict[object, List[np.ndarray]] = {}
        # seq_id -> current step's [L, K] being filled
        self._cur: Dict[object, np.ndarray] = {}
        self._cur_filled: Dict[object, int] = {}

    def start_step(self, seq_id) -> None:
        """Flush any in-progress step and begin a fresh one for this seq."""
        self._flush(seq_id)
        self._cur[seq_id] = np.full((self.meta.L, self.meta.K), -1, dtype=np.int64)
        self._cur_filled[seq_id] = 0

    def record(self, seq_id, layer_id: int, expert_ids) -> None:
        cur = self._cur.get(seq_id)
        if cur is None:
            self.start_step(seq_id)
            cur = self._cur[seq_id]
        ids = np.asarray(expert_ids, dtype=np.int64).ravel()
        k = min(self.meta.K, ids.shape[0])
        cur[layer_id, :k] = ids[:k]
        if k < self.meta.K:        # pad short rows by repeating last id
            cur[layer_id, k:] = ids[k - 1] if k > 0 else 0

    def _flush(self, seq_id) -> None:
        cur = self._cur.pop(seq_id, None)
        self._cur_filled.pop(seq_id, None)
        if cur is not None:
            self._buf.setdefault(seq_id, []).append(cur)

    def finish(self, seq_id, ordinal: int) -> Optional[RequestData]:
        self._flush(seq_id)
        steps = self._buf.pop(seq_id, None)
        if not steps:
            return None
        sel = np.stack(steps, axis=0)       # [T, L, K]
        return RequestData(T=sel.shape[0], sel=sel, ord=int(ordinal))

    def drop(self, seq_id) -> None:
        self._buf.pop(seq_id, None)
        self._cur.pop(seq_id, None)
        self._cur_filled.pop(seq_id, None)


# ===================================================================
# Provider: aggregate per-request predictions -> A2Hotness / A2Demand
# ===================================================================
class A2HotnessProvider:
    """Holds the per-step replicated A2 matrices and exposes the two providers.

    Per decode step the runtime calls ``set_step`` with, for each ACTIVE request,
    its [L, E] future-hotness prediction and owning rank.  This builds:
        A2Hotness[L, E]    = sum over all active requests of pred
        A2Demand[L, E, G]  = sum over requests on rank g of pred
    For multi-rank determinism, pass ``all_reduce_fn`` to sum each rank's local
    matrices into the global (identical-on-every-rank) result.
    """

    def __init__(self, meta: A2Meta, ep_size: int):
        self.meta = meta
        self.ep_size = ep_size
        self.hotness = np.zeros((meta.L, meta.E), dtype=np.float64)
        self.demand = np.zeros((meta.L, meta.E, ep_size), dtype=np.float64)

    def set_step(
        self,
        preds: List[Tuple[np.ndarray, int]],
        all_reduce_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """preds: list of (pred[L,E] float64, owner_rank) for active requests on
        THIS rank.  ``all_reduce_fn`` (optional) sums an array across EP ranks so
        the resulting matrices are identical everywhere."""
        L, E, G = self.meta.L, self.meta.E, self.ep_size
        demand = np.zeros((L, E, G), dtype=np.float64)
        for pred, rank in preds:
            if pred is None:
                continue
            demand[:, :, rank] += pred
        if all_reduce_fn is not None:
            demand = all_reduce_fn(demand)
        self.demand = demand
        self.hotness = demand.sum(axis=2)

    def clear(self) -> None:
        self.hotness.fill(0.0)
        self.demand.fill(0.0)

    # --- the two callables OursPlacementPlanner consumes ---
    def evict_cost(self, key: Tuple[int, int]) -> float:
        l, e = key
        if 0 <= l < self.meta.L and 0 <= e < self.meta.E:
            return float(self.hotness[l, e])
        return 0.0

    def future_affinity(self, key: Tuple[int, int], rank: int) -> float:
        l, e = key
        if 0 <= l < self.meta.L and 0 <= e < self.meta.E and 0 <= rank < self.ep_size:
            return float(self.demand[l, e, rank])
        return 0.0


# ===================================================================
# Frequency (past usage) + the evict-cost combiner  evict_cost = Future·(1+μ·NormFreq)
# ===================================================================
class GlobalFreqTracker:
    """Bounded per-(layer, expert) frequency = "지금까지 얼마나 쓰였나" (past).

    FineMoE-style **bounded window**: keep only the last ``capacity`` completed
    requests' per-(l,e) usage; older ones are subtracted out (ring buffer).  Each
    request contributes a ``[L, E]`` count matrix (how many times it used each
    (l,e)).  ``norm_freq`` returns per-layer max-normalised freq in [0, 1].
    """

    def __init__(self, meta: A2Meta, capacity: int = 1000):
        self.meta = meta
        self.capacity = int(capacity)
        self.freq = np.zeros((meta.L, meta.E), dtype=np.float64)  # running bounded sum
        self._ring: Deque[np.ndarray] = deque()

    def add_request(self, req_counts: np.ndarray) -> None:
        """req_counts: [L, E] usage counts of ONE completed request."""
        req_counts = np.asarray(req_counts, dtype=np.float64)
        self.freq += req_counts
        self._ring.append(req_counts)
        while len(self._ring) > self.capacity:
            old = self._ring.popleft()
            self.freq -= old
        np.clip(self.freq, 0.0, None, out=self.freq)  # guard fp drift

    def norm_freq(self) -> np.ndarray:
        """[L, E] freq / max_e freq  (per-layer), in [0, 1]."""
        m = self.freq.max(axis=1, keepdims=True)
        out = np.zeros_like(self.freq)
        nz = m[:, 0] > 0
        out[nz] = self.freq[nz] / m[nz]
        return out


class EvictCostProvider:
    """evict_cost(l,e) = Future_layer(l,e) · (1 + μ · NormFreq(l,e)).

    - **Future_layer**: per-(l,e) "앞으로 쓸까" 예측 [L,E] (EAMC layer-축 또는 A2);
      ``set_future`` 로 매 step/layer 주입.  source-agnostic.
    - **NormFreq**: :class:`GlobalFreqTracker` 의 max-정규화 빈도 (과거 보정).
    - cold-start (future 미주입/무효) → NormFreq 로 fallback (≈ LFU-by-freq).

    ``ours_planner`` 의 ``evict_cost_provider`` 로 ``.evict_cost`` 메서드를 그대로 꽂는다.
    """

    def __init__(self, meta: A2Meta, mu: float = 0.5,
                 freq_tracker: Optional[GlobalFreqTracker] = None):
        self.meta = meta
        self.mu = float(mu)
        self.freq = freq_tracker
        self._future: Optional[np.ndarray] = None   # [L, E]
        self._normfreq: Optional[np.ndarray] = None

    def set_future(self, future_LE: Optional[np.ndarray]) -> None:
        """Inject this step's Future_layer [L,E] (None → cold-start fallback)."""
        self._future = None if future_LE is None else np.asarray(
            future_LE, dtype=np.float64)
        self._normfreq = self.freq.norm_freq() if self.freq is not None else None

    def evict_cost(self, key: Tuple[int, int]) -> float:
        l, e = key
        if not (0 <= l < self.meta.L and 0 <= e < self.meta.E):
            return 0.0
        nf = float(self._normfreq[l, e]) if self._normfreq is not None else 0.0
        if self._future is None:
            return nf                          # cold-start: freq-only (LFU-ish)
        fut = float(self._future[l, e])
        return fut * (1.0 + self.mu * nf)      # Future × (1 + μ·NormFreq)


# ===================================================================
# self-[L,E] hotness — running accumulated demand (no history retrieval)
# ===================================================================
class RunningDemandHotness:
    """Deterministic self-[L,E] evict-cost for OURS (no trajectory retrieval).

    hotness = running accumulated per-(layer, expert) demand counts, built from the
    all-gathered demand the controller already computes each layer -> IDENTICAL on
    every rank -> drift-safe.  Decode is stationary, so "demanded a lot so far"
    predicts "demanded a lot soon".

        NormFreq[l,e]   = Accum[l,e] / max_e Accum[l,·]          (per-layer, [0,1])
        evict_cost(l,e) = Accum[l,e] · (1 + μ · NormFreq[l,e])

    Unlike the LFU stub (freq≈1 after warmup -> flat -> quota DP degenerate), Accum
    is hot≫cold so the decode quota DP gets genuine cost variation across GPUs.
    """

    def __init__(self, num_layers: int, num_experts: int, mu: float = 0.5):
        self.L = int(num_layers)
        self.E = int(num_experts)
        self.mu = float(mu)
        self.accum = np.zeros((self.L, self.E), dtype=np.float64)
        self._norm: Optional[np.ndarray] = None   # cached normfreq, invalidated on update

    def update(self, demand: Dict[Tuple[int, int], int]) -> None:
        """Add one processed layer/step's demand.  demand: {(layer, expert): count}."""
        for (l, e), c in demand.items():
            if 0 <= l < self.L and 0 <= e < self.E:
                self.accum[l, e] += c
        self._norm = None

    def _normfreq(self) -> np.ndarray:
        if self._norm is None:
            m = self.accum.max(axis=1, keepdims=True)
            out = np.zeros_like(self.accum)
            nz = m[:, 0] > 0
            out[nz] = self.accum[nz] / m[nz]
            self._norm = out
        return self._norm

    def evict_cost(self, key: Tuple[int, int]) -> float:
        l, e = key
        if not (0 <= l < self.L and 0 <= e < self.E):
            return 0.0
        nf = float(self._normfreq()[l, e])
        return float(self.accum[l, e] * (1.0 + self.mu * nf))


# ===================================================================
# [L,E] retrieval hotness — per-request history collection + cosine top-R
# ===================================================================
def _norm_perlayer(M: np.ndarray) -> np.ndarray:
    """[L,E] -> per-layer routing probability (row sum = 1; zero rows stay 0)."""
    M = np.asarray(M, dtype=np.float64)
    s = M.sum(axis=1, keepdims=True)
    out = np.zeros_like(M)
    nz = s[:, 0] > 0
    out[nz] = M[nz] / s[nz]
    return out


class RankDemandAccumulator:
    """Per-rank running demand `[G, L, E]` from the all-gathered ``per_rank_count``
    (identical on every rank -> deterministic).  Reset per eval-iteration (a new
    batch of requests).  With batch_per_rank==1 each rank's matrix == one request's
    [L,E] (used to build the per-request collection)."""

    def __init__(self, ep_size: int, num_layers: int, num_experts: int):
        self.G, self.L, self.E = int(ep_size), int(num_layers), int(num_experts)
        self.accum = np.zeros((self.G, self.L, self.E), dtype=np.float64)

    def update(self, layer_id: int, per_rank_counts) -> None:
        if 0 <= layer_id < self.L:
            self.accum[:, layer_id, :] += np.asarray(per_rank_counts, dtype=np.float64)

    def reset(self) -> None:
        self.accum.fill(0.0)

    def rank_matrix(self, g: int) -> np.ndarray:
        return self.accum[g].copy()

    def all(self) -> np.ndarray:
        return self.accum.copy()


class RetrievalHotnessProvider:
    """history `[L,E]` retrieval (EAMC-style) -> evict_cost + future_affinity.

    - collection: per-request (or per-rank-batch) `[L,E]`, per-layer normalised.
    - per decode step ``refresh(rank_accum[G,L,E])``: each rank g queries with its
      normalised `[L,E]`, cosine **top-R** over the collection -> averaged match
      ``pred_g``.  Then:
          evict_cost(l,e)      = normalize_perlayer( Σ_g pred_g )[l,e]   (GLOBAL P)
          future_affinity(l,e,g)= pred_g[l,e]                            (per-rank)
    - ``add`` appends completed batches online (cap).  cosine top-1 scores logged
      so discriminability is measurable (all-high == undiscriminative).
    - Deterministic: collection identical across ranks; query from all-gathered
      RankAccum -> identical pred -> drift-safe.
    """

    def __init__(self, ep_size: int, num_layers: int, num_experts: int,
                 topR: int = 1, cap: int = 1000):
        self.G, self.L, self.E = int(ep_size), int(num_layers), int(num_experts)
        self.R = int(topR)
        self.cap = int(cap)
        self.col: List[np.ndarray] = []          # per-layer-normalised [L,E]
        self._flat = None                        # [N, L*E]
        self._fnorm = None                       # [N]
        self._evict = np.zeros((self.L, self.E), dtype=np.float64)
        self._aff = np.zeros((self.L, self.E, self.G), dtype=np.float64)
        self.cos_top1: List[float] = []          # logged top-1 cosine per (step, rank)

    # ---- collection management ----
    def load_collection(self, mats) -> None:
        self.col = [_norm_perlayer(m) for m in mats]
        self._rebuild()

    def add(self, mat) -> None:
        self.col.append(_norm_perlayer(mat))
        while len(self.col) > self.cap:
            self.col.pop(0)
        self._rebuild()

    def _rebuild(self) -> None:
        if self.col:
            self._flat = np.stack([m.ravel() for m in self.col], axis=0)
            self._fnorm = np.linalg.norm(self._flat, axis=1) + 1e-12
        else:
            self._flat = None

    # ---- per-step refresh (cosine retrieval) ----
    def refresh(self, rank_accum: np.ndarray) -> None:
        if self._flat is None:                   # cold: fall back to self
            self._evict = _norm_perlayer(rank_accum.sum(axis=0))
            for g in range(self.G):
                self._aff[:, :, g] = _norm_perlayer(rank_accum[g])
            return
        preds = []
        for g in range(self.G):
            q = _norm_perlayer(rank_accum[g]).ravel()
            qn = np.linalg.norm(q) + 1e-12
            sims = (self._flat @ q) / (self._fnorm * qn)        # [N]
            if self.R <= 1:
                best = int(np.argmax(sims))                     # tie -> lowest idx
                self.cos_top1.append(float(sims[best]))
                preds.append(self.col[best])
            else:
                R = min(self.R, len(self.col))
                top = np.argpartition(sims, len(sims) - R)[len(sims) - R:]
                top = top[np.argsort(sims[top])[::-1]]
                self.cos_top1.append(float(sims[top[0]]))
                preds.append(_norm_perlayer(sum(self.col[int(i)] for i in top)))
        for g in range(self.G):
            self._aff[:, :, g] = preds[g]
        self._evict = _norm_perlayer(sum(preds))

    # ---- the two callables ours_planner consumes ----
    def evict_cost(self, key: Tuple[int, int]) -> float:
        l, e = key
        return float(self._evict[l, e]) if (0 <= l < self.L and 0 <= e < self.E) else 0.0

    def future_affinity(self, key: Tuple[int, int], rank: int) -> float:
        l, e = key
        if 0 <= l < self.L and 0 <= e < self.E and 0 <= rank < self.G:
            return float(self._aff[l, e, rank])
        return 0.0
