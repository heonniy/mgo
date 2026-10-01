"""Unit tests that don't require moe_infinity or NCCL to be installed."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make repo importable without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_infinity_ep.launch.distributed_setup import ProcessTopology, pin_visible_device  # noqa: E402
from moe_infinity_ep.utils.config import Config  # noqa: E402


def test_pin_visible_device_sets_env(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rv = pin_visible_device()
    assert rv == 3
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_topology_unified_ep():
    # Unified EP: world == ep_size, ep_rank == global_rank, no separate dp.
    topo = ProcessTopology(world_size=2, ep_size=2, global_rank=0, ep_rank=0, local_device=0)
    assert topo.is_rank0
    assert topo.ep_rank == topo.global_rank

    topo = ProcessTopology(world_size=2, ep_size=2, global_rank=1, ep_rank=1, local_device=0)
    assert not topo.is_rank0


def test_config_loads_default(tmp_path: Path):
    yaml_text = """
parallel: {ep_size: 0}
model: {path: /nonexistent, arch: qwen3_moe}
offload: {path: /tmp/x}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = Config.from_yaml(p)
    assert cfg.parallel.ep_size == 0
    assert cfg.model.arch == "qwen3_moe"
    assert cfg.offload.device_memory_ratio == 0.75  # default


def test_config_resolve_auto():
    cfg = Config()
    cfg.parallel.ep_size = 0
    cfg.resolve_parallel(world_size=2)
    assert cfg.parallel.ep_size == 2


def test_config_resolve_mismatch_raises():
    cfg = Config()
    cfg.parallel.ep_size = 4
    with pytest.raises(ValueError, match="unified EP requires"):
        cfg.resolve_parallel(world_size=2)
