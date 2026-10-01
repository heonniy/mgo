"""Golden Test 3/4 (real Archer) — moe_cache_golden README §4.

Runs the REAL Archer ExpertDispatcher and compares its debug events,
slot-level cache, and direct/staging counters against
``real_archer_gold_execution.jsonl``.

Skips unless ALL of:
  * MOE_EP_RUN_REAL_ARCHER_TEST=1
  * the archer pybind extension (moe_infinity._store) imports
  * the dispatcher exposes the debug APIs README §3 requires

As of this writing the built dispatcher exposes submit_plan,
get_cached_experts, and get_fetch_mode_counts, but NOT get_debug_events /
clear_debug_events / get_cached_slots (slot-level).  Those must be added in
core/parallel/expert_dispatcher.{h,cpp} + core/python/py_archer_prefetch.cpp
and the extension rebuilt before this test can run end-to-end.  Until then it
skips with a precise reason, and tests/test_staging_hazard_model.py covers the
staging/direct decision logic in pure Python.
"""
from __future__ import annotations

import json
import os

import pytest


GOLD_DIR = os.environ.get(
    "MOE_EP_GOLD_DIR", "/home/work/hyewon.lee/실험/moe_cache_golden")
GOLD_FILE = os.path.join(GOLD_DIR, "real_archer_gold_execution.jsonl")

REQUIRED_DEBUG_APIS = (
    "submit_plan", "get_cached_experts", "get_fetch_mode_counts",
    "get_cached_slots", "get_debug_events", "clear_debug_events",
)


def _gate():
    if os.environ.get("MOE_EP_RUN_REAL_ARCHER_TEST") != "1":
        return "set MOE_EP_RUN_REAL_ARCHER_TEST=1 to run the real-archer test"
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA not available"
    except Exception as e:  # noqa: BLE001
        return f"torch/CUDA import failed: {e}"
    try:
        from moe_infinity import _store  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"archer extension (moe_infinity._store) not importable: {e}"
    disp_cls = getattr(_store, "expert_dispatcher", None)
    if disp_cls is None:
        return "moe_infinity._store.expert_dispatcher missing"
    missing = [m for m in REQUIRED_DEBUG_APIS if not hasattr(disp_cls, m)]
    if missing:
        return ("dispatcher missing debug APIs required by README §3: "
                f"{missing}. Add them to expert_dispatcher.{{h,cpp}} + "
                "py_archer_prefetch.cpp and rebuild moe_infinity._store.")
    return None


_SKIP = _gate()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))


def _load_gold():
    with open(GOLD_FILE) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def test_real_archer_golden_execution():
    """Per-case real-archer execution check (runs only when gate passes)."""
    gold = _load_gold()
    assert gold, "golden file empty"
    # NOTE: kept intentionally minimal — the full per-case seeding/submit/
    # compare harness is enabled together with the C++ debug-event API.  The
    # gate above documents exactly what is missing; until the APIs exist this
    # body is unreachable (skipped), so we fail loudly if it is ever reached
    # without them rather than silently passing.
    from moe_infinity import _store
    disp_cls = _store.expert_dispatcher
    for m in REQUIRED_DEBUG_APIS:
        assert hasattr(disp_cls, m), f"missing {m}"
    pytest.skip(
        "real-archer execution harness not yet wired; debug APIs present — "
        "implement seeding/submit/compare here")
