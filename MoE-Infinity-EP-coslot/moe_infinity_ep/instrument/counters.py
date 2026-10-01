"""Counters — always-on per-rank metrics, written from EPExpertExecutor.

These are the measurements we feed into policy-vs-policy ablations. Counters
are policy-agnostic: every policy (naive or smart) updates the same fields.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Cap per-list growth to defend against unbounded memory pressure on long
# runs.  Each entry is small (tuple of ints / small dict), so 1M entries
# is ~80MB worst case.  Override via env when capturing a full warm-up.
_EXPERT_ACCESS_CAP = int(os.environ.get("MOE_EP_EXPERT_ACCESS_TRACE_CAP",
                                        "1000000") or 1000000)
_ROUTING_LOG_CAP = int(os.environ.get("MOE_EP_ROUTING_LOG_CAP",
                                      "1000000") or 1000000)

# M15 fix (2026-05-27): generic cap for the per-layer latency / counter lists
# (layer_total_us, cache_sync_us, a2a_*_us, local_exec_us, dispatch_*_us,
# layer_hits/misses/evictions/remote_serves, prefill/decode mirrors,
# drift_layer_pre, cap_utilization_max, etc).  Previously these grew
# unbounded — on long daemon-style runs the 15+ lists at 47 layers per
# forward × 100K forwards × 8B ≈ 560 MB / rank.  Drop-on-full keeps the
# bound predictable.  Override via env to capture a full long run trace.
_DEFAULT_LIST_CAP = int(os.environ.get("MOE_EP_COUNTERS_LIST_CAP",
                                       "5000000") or 5000000)


def _capped_append(lst, value, cap: int = _DEFAULT_LIST_CAP) -> None:
    """Append unless ``lst`` is at cap. No-op on overflow (don't fail the
    measurement just because the bookkeeping is full)."""
    if len(lst) >= cap:
        return
    lst.append(value)


def _capped_extend(lst, values, cap: int = _DEFAULT_LIST_CAP) -> None:
    """Extend up to ``cap`` and silently drop the tail beyond."""
    remaining = cap - len(lst)
    if remaining <= 0:
        return
    if remaining >= len(values):
        lst.extend(values)
    else:
        lst.extend(list(values)[:remaining])


@dataclass
class Counters:
    # --- bytes moved ---
    pcie_fetch_bytes: int = 0
    # Number of cross-rank expert migrations applied (PlacementOp with
    # source="nvlink" and source_rank != target_rank).
    cross_rank_migrations: int = 0
    # Total archer.explicit_fetch()/enqueue_prefetch() calls.  Surfaced by
    # the executor from CommandDispatcher.n_explicit_prefetch.
    prefetch_issued_count: int = 0
    # Surface CommandDispatcher's failure counters.  Non-zero values mean
    # the controller's plan didn't fully land in archer — drift source.
    dispatch_fetch_failures: int = 0
    dispatch_evict_failures: int = 0
    dispatch_evict_noops: int = 0
    # Cap-undersize indicator from eviction_planner.  Non-zero = working
    # set exceeded cap_per_rank and some install_ops couldn't be paired
    # with an evict_op → cache temporarily over-runs cap.  Raise cap or
    # reduce demand.
    evict_no_victim_count: int = 0
    # EAMC priority counters (opt-in via MOE_EP_EAMC_PRIORITY=1).
    # n_predicts: heavy cosine-similarity calls actually executed
    # n_updates: cheap tracer.update_entry calls (every layer when enabled)
    # n_skipped_stride: layers where stride caused predict to be skipped
    eamc_n_predicts: int = 0
    eamc_n_updates: int = 0
    eamc_n_skipped_stride: int = 0
    nvlink_expert_migration_bytes: int = 0
    nvlink_activation_bytes: int = 0      # tokens routed through all-to-all

    # --- cache events ---
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    cache_remote_serves: int = 0          # hit found on another rank

    # --- per-rank work ---
    tokens_processed: int = 0             # local input tokens (sum across forwards)
    layers_executed: int = 0

    # --- per-layer timing (microseconds, appended each call) ---
    layer_total_us: List[int] = field(default_factory=list)
    cache_sync_us: List[int] = field(default_factory=list)
    policy_us: List[int] = field(default_factory=list)
    a2a_forward_us: List[int] = field(default_factory=list)
    local_exec_us: List[int] = field(default_factory=list)
    a2a_backward_us: List[int] = field(default_factory=list)

    # --- archer-state ↔ cache_view drift per layer (length = layers executed) ---
    # Populated by EPExpertExecutor from controller.last_drift_pre each layer.
    # 0 = controller and archer agree.  Non-zero = either an autonomous archer
    # mutation slipped through or our explicit_evict/fetch didn't land.
    drift_layer_pre: List[int] = field(default_factory=list)

    # --- 2026-05-27 instrumentation upgrade ---
    # Per-call latency of archer.explicit_fetch / explicit_evict in microseconds.
    # Aggregated across all layers — distribution analysis (p50/p99) in summary.
    # Production rate expected:
    #   * explicit_fetch: ~10ms PCIe sync copy (Qwen3-235B per expert).
    #   * explicit_evict: <1ms (GPU free).
    # Sustained outliers (e.g. p99 > 50ms) suggest disk-pressure or NVLink
    # contention; spikes ≫ p99 are usually GPUFetchFunc fallback fetches that
    # missed our explicit_fetch (i.e. drift indicator).
    dispatch_fetch_us:  List[int] = field(default_factory=list)
    dispatch_evict_us:  List[int] = field(default_factory=list)

    # Max cap utilization across ranks per layer (0.0 .. 1.0).  Persisted per
    # layer so we can chart pressure evolution and correlate with evict bursts.
    cap_utilization_max: List[float] = field(default_factory=list)
    # Count of layers where at least one rank had util >= 0.9.  Sustained
    # non-zero → working set approaching cap; consider raising sparse_hbm_ratio.
    high_cap_pressure_layers: int = 0
    # Count of layers where every rank had util >= 0.95.  Drift FATAL is near.
    near_cap_full_layers: int = 0

    # Bounded log of drift events: (step, layer_id, rank, drift_pre, evicted,
    # installed).  Capped at 1000 entries — drift in steady state should be 0,
    # so a long log means a correctness issue worth surfacing.  See controller's
    # _sync_archer_state for the source.
    drift_event_log: List[Tuple[int, int, int, int, int, int]] = field(
        default_factory=list,
    )

    # Number of times `command_dispatcher.execute` saw a non-trivial sub-batch
    # of evict_failures > 0 OR fetch_failures > 0 — these surface drift between
    # the Python controller's intent and what landed in archer.  Indexes into
    # `dispatch_failure_examples` for full context (first 16).
    dispatch_failure_examples: List[dict] = field(default_factory=list)

    # --- per-layer cache events (appended each multi-rank dispatch) ---
    layer_hits: List[int] = field(default_factory=list)
    layer_misses: List[int] = field(default_factory=list)
    layer_evictions: List[int] = field(default_factory=list)
    layer_remote_serves: List[int] = field(default_factory=list)

    # --- per-phase timing (microseconds) — keyed off `is_decode`. Phase-split
    # mirrors of the always-on timers above, used by the main benchmark to
    # report prefill vs decode breakdowns separately (Plan §2 phase routing).
    prefill_layer_total_us:  List[int] = field(default_factory=list)
    prefill_cache_sync_us:   List[int] = field(default_factory=list)
    prefill_a2a_forward_us:  List[int] = field(default_factory=list)
    prefill_local_exec_us:   List[int] = field(default_factory=list)
    prefill_a2a_backward_us: List[int] = field(default_factory=list)
    decode_layer_total_us:   List[int] = field(default_factory=list)
    decode_cache_sync_us:    List[int] = field(default_factory=list)
    decode_a2a_forward_us:   List[int] = field(default_factory=list)
    decode_local_exec_us:    List[int] = field(default_factory=list)
    decode_a2a_backward_us:  List[int] = field(default_factory=list)

    # --- per-phase counters (scalars) ---
    prefill_tokens_processed: int = 0
    decode_tokens_processed:  int = 0
    prefill_cache_hits:       int = 0
    prefill_cache_misses:     int = 0
    prefill_cache_evictions:  int = 0
    decode_cache_hits:        int = 0
    decode_cache_misses:      int = 0
    decode_cache_evictions:   int = 0

    # --- per-sample latency (filled by the harness, not the executor) ---
    # One entry per (rank, sample) pair. Aggregator concatenates across ranks
    # and reports p50/p95 in summary.csv.
    ttft_us:                  List[int] = field(default_factory=list)
    tpot_us:                  List[int] = field(default_factory=list)
    e2e_us:                   List[int] = field(default_factory=list)
    prefill_tokens_per_sample: List[int] = field(default_factory=list)
    decode_tokens_per_sample:  List[int] = field(default_factory=list)

    # --- expert access trace: (step, layer, ep_rank, expert_id, n_tokens) ---
    expert_access_trace: List[Tuple[int, int, int, int, int]] = field(
        default_factory=list,
    )

    # --- per-(step, layer) routing log (env MOE_EP_ROUTING_LOG=1) ---
    # Each entry captures the policy decision + cache state at one MoE layer
    # call. Used to cross-check that naive/balanced actually diverge in
    # per-rank miss distribution, and to validate decode hit-rate sanity.
    # Schema:
    #   step, layer, is_decode,
    #   n_demanded, n_hits, n_misses,
    #   hits_per_serving_rank   : list[int] of len ep_size
    #   miss_target_rank_dist   : list[int] of len ep_size
    #   miss_expert_ids         : list[int]
    #   my_rank_cache_size      : int (this rank's cache slot usage at entry)
    #   global_unique_experts   : int (sum of unique (l,e) across all ranks)
    routing_log: List[dict] = field(default_factory=list)

    # --- internal step counter (advanced by the executor each forward) ---
    _step: int = 0

    # --- archer-side ground-truth cache stats snapshot ---
    # Populated by run_ours_ep harness right before gather_and_dump, from
    # ``archer_engine.expert_dispatcher.get_cache_stats()``. Cross-checks
    # cache_view↔archer drift hypothesis (B). Fields mirror the C++ tensor
    # return order in expert_dispatcher.cpp:GetCacheStats().
    archer_visit_cnt: int = 0
    archer_hit_cnt: int = 0
    archer_miss_cnt: int = 0
    archer_gpu_fetch_cnt: int = 0
    archer_eviction_cnt: int = 0
    archer_overload_fetch_cnt: int = 0

    # Per-forward archer snapshot — populated by EPExpertExecutor at the start
    # of each new forward (advance_step). Each entry is the cumulative archer
    # state at that point. Diff between consecutive entries = per-forward delta.
    # Enable via env MOE_EP_ARCHER_PERFWD_SNAPSHOT=1 (cost ~0.5 ms per forward
    # due to .cpu().tolist() sync, but worth it for drift evolution).
    archer_snapshots: List[Tuple[int, int, int, int, int, int, int]] = field(
        default_factory=list,
    )

    # ---- API ----

    def bump(self, name: str, val: int = 1) -> None:
        setattr(self, name, getattr(self, name) + val)

    @contextmanager
    def time(self, name: str):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            # Never let timer bookkeeping mask the original exception (NCCL
            # error, CUDA OOM, etc).  Best-effort append; on lookup failure
            # silently skip — the assertion-on-missing pattern used to swallow
            # the in-flight exception via "during handling of another exception".
            dt_us = (time.perf_counter_ns() - t0) // 1000
            lst = getattr(self, name, None)
            if isinstance(lst, list):
                _capped_append(lst, int(dt_us))

    @contextmanager
    def time_phased(self, name_base: str, is_decode: bool):
        """Append the measured µs to ``prefill_<name_base>`` or
        ``decode_<name_base>``.  Same exception-safe pattern as ``time``.
        """
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            dt_us = (time.perf_counter_ns() - t0) // 1000
            attr = ("decode_" if is_decode else "prefill_") + name_base
            lst = getattr(self, attr, None)
            if isinstance(lst, list):
                _capped_append(lst, int(dt_us))

    def log_expert_access(
        self, layer_id: int, ep_rank: int, expert_id: int, n_tokens: int,
    ) -> None:
        # Drop on overflow rather than blowing past memory cap silently.
        if len(self.expert_access_trace) >= _EXPERT_ACCESS_CAP:
            return
        self.expert_access_trace.append(
            (self._step, int(layer_id), int(ep_rank), int(expert_id), int(n_tokens))
        )

    def log_routing(self, entry: dict) -> None:
        """Append one routing-decision record. Guarded by env in the caller
        (MOE_EP_ROUTING_LOG=1) so production runs incur zero overhead."""
        if len(self.routing_log) >= _ROUTING_LOG_CAP:
            return
        self.routing_log.append(entry)

    def advance_step(self) -> None:
        self._step += 1

    def to_dict(self) -> Dict:
        return {
            "pcie_fetch_bytes": self.pcie_fetch_bytes,
            "cross_rank_migrations": self.cross_rank_migrations,
            "prefetch_issued_count": self.prefetch_issued_count,
            "dispatch_fetch_failures": self.dispatch_fetch_failures,
            "dispatch_evict_failures": self.dispatch_evict_failures,
            "dispatch_evict_noops":    self.dispatch_evict_noops,
            "evict_no_victim_count":   self.evict_no_victim_count,
            "eamc_n_predicts":         self.eamc_n_predicts,
            "eamc_n_updates":          self.eamc_n_updates,
            "eamc_n_skipped_stride":   self.eamc_n_skipped_stride,
            "nvlink_expert_migration_bytes": self.nvlink_expert_migration_bytes,
            "nvlink_activation_bytes": self.nvlink_activation_bytes,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_evictions": self.cache_evictions,
            "cache_remote_serves": self.cache_remote_serves,
            "tokens_processed": self.tokens_processed,
            "layers_executed": self.layers_executed,
            "layer_total_us_sum": sum(self.layer_total_us),
            "cache_sync_us_sum": sum(self.cache_sync_us),
            "policy_us_sum": sum(self.policy_us),
            "a2a_forward_us_sum": sum(self.a2a_forward_us),
            "local_exec_us_sum": sum(self.local_exec_us),
            "a2a_backward_us_sum": sum(self.a2a_backward_us),
            "drift_layer_pre_sum": sum(self.drift_layer_pre),
            # phase-split scalars
            "prefill_tokens_processed": self.prefill_tokens_processed,
            "decode_tokens_processed":  self.decode_tokens_processed,
            "prefill_cache_hits":       self.prefill_cache_hits,
            "prefill_cache_misses":     self.prefill_cache_misses,
            "prefill_cache_evictions":  self.prefill_cache_evictions,
            "decode_cache_hits":        self.decode_cache_hits,
            "decode_cache_misses":      self.decode_cache_misses,
            "decode_cache_evictions":   self.decode_cache_evictions,
            # phase-split timer sums
            "prefill_layer_total_us_sum":  sum(self.prefill_layer_total_us),
            "prefill_cache_sync_us_sum":   sum(self.prefill_cache_sync_us),
            "prefill_a2a_forward_us_sum":  sum(self.prefill_a2a_forward_us),
            "prefill_local_exec_us_sum":   sum(self.prefill_local_exec_us),
            "prefill_a2a_backward_us_sum": sum(self.prefill_a2a_backward_us),
            "decode_layer_total_us_sum":   sum(self.decode_layer_total_us),
            "decode_cache_sync_us_sum":    sum(self.decode_cache_sync_us),
            "decode_a2a_forward_us_sum":   sum(self.decode_a2a_forward_us),
            "decode_local_exec_us_sum":    sum(self.decode_local_exec_us),
            "decode_a2a_backward_us_sum":  sum(self.decode_a2a_backward_us),
            # per-sample latency lists (raw — aggregator computes p50/p95)
            "ttft_us":                    list(self.ttft_us),
            "tpot_us":                    list(self.tpot_us),
            "e2e_us":                     list(self.e2e_us),
            "prefill_tokens_per_sample":  list(self.prefill_tokens_per_sample),
            "decode_tokens_per_sample":   list(self.decode_tokens_per_sample),
            "expert_access_trace_len":    len(self.expert_access_trace),
            "routing_log_len":            len(self.routing_log),
            # archer-state drift per layer (zero in steady state)
            "drift_layer_pre":            list(self.drift_layer_pre),
            "current_step":               self._step,
            # 2026-05-27 instrumentation upgrades
            "dispatch_fetch_us":          list(self.dispatch_fetch_us),
            "dispatch_evict_us":          list(self.dispatch_evict_us),
            "dispatch_fetch_us_sum":      sum(self.dispatch_fetch_us),
            "dispatch_evict_us_sum":      sum(self.dispatch_evict_us),
            "cap_utilization_max":        list(self.cap_utilization_max),
            "high_cap_pressure_layers":   self.high_cap_pressure_layers,
            "near_cap_full_layers":       self.near_cap_full_layers,
            "drift_event_log":            list(self.drift_event_log),
            "dispatch_failure_examples":  list(self.dispatch_failure_examples),
            # archer ground truth (snapshot before gather_and_dump)
            "archer_visit_cnt":           self.archer_visit_cnt,
            "archer_hit_cnt":             self.archer_hit_cnt,
            "archer_miss_cnt":            self.archer_miss_cnt,
            "archer_gpu_fetch_cnt":       self.archer_gpu_fetch_cnt,
            "archer_eviction_cnt":        self.archer_eviction_cnt,
            "archer_overload_fetch_cnt":  self.archer_overload_fetch_cnt,
            # per-forward archer evolution
            "archer_snapshots":           [list(s) for s in self.archer_snapshots],
        }
