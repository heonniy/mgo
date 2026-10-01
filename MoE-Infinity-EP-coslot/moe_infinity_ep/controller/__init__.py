"""Centralized Python cache controller for multi-process MoE EP+DP.

Replaces the previous decentralized (cache_view + fetch_policy + archer LFU)
hybrid with a single source of truth:

  Python Controller decides ALL cache state (install / evict / prefetch / pin).
  archer = rank-local GPU storage executor (no autonomous policy).

See ``GlobalCacheController`` for the orchestration entry point.
"""
from .global_controller import GlobalCacheController  # noqa: F401
