"""YAML config loader and config dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ParallelConfig:
    # Unified EP convention: every rank participates in one EP group of size
    # world_size, and each rank holds its own batch shard (implicit DP = world).
    # No separate DP slabs. ep_size is just for validation; 0 = auto = world.
    ep_size: int = 0


@dataclass
class ModelConfig:
    path: str = ""
    arch: str = "qwen3_moe"


@dataclass
class OffloadConfig:
    path: str = "/tmp/moe_ep_offload"
    # Per-GPU device memory budget for archer's overall pool (dense + sparse
    # + workspace).  Propagated via ArcherConfig.device_memory_ratio →
    # archer_prefetch_handle ctor (model_offload.py:174).  ALWAYS WIRED.
    device_memory_ratio: float = 0.75
    # Per-process host (CPU) RAM budget for archer's HostMemoryPool.
    # Propagated via MOE_INFINITY_HOST_MEMORY_RATIO env in
    # distributed_engine._sync_archer_host_ratio_env (Round 3 fix; the
    # ArcherConfig field alone is silently inert because the C++ side only
    # reads the env var).
    host_memory_ratio: float = 0.9
    # GlobalCacheView capacity per rank, in (layer, expert) slots.
    # 0 = derived from sparse_hbm_ratio below (or fall back to unlimited if
    #     both are 0). Set non-zero to override the auto-calc.
    cache_capacity_per_rank: int = 0
    # Sparse-cache budget as a fraction of per-GPU HBM. When > 0 and
    # cache_capacity_per_rank == 0, cap auto-derives as:
    #   cap = floor((HBM_per_gpu × sparse_hbm_ratio) / per_expert_bytes)
    # where per_expert_bytes = 3 × intermediate × hidden × dtype_bytes.
    # Independent of GPU count (each rank gets the same cap).
    # Round 3 fix: cap × per_expert_bytes is also exported as
    # MOE_INFINITY_SPARSE_BYTES so archer's GetSparseCacheLimit returns the
    # exact same byte budget as Python's cap.  Without this, archer's
    # MOE_INFINITY_SPARSE_RATIO env (default 0.4) was the actual signal —
    # an independent knob that drifted when only the YAML field was changed.
    sparse_hbm_ratio: float = 0.0
    # Path to a previously-saved EAMC (ExpertTracer.trace_collection .npy).
    # If set, archer auto-loads via ArcherConfig.trace_path → OffloadEngine
    # init. Used by the main_exp sweep to reuse a warm-up's expert access
    # matrix across all measurement runs for the same (model, dataset).
    trace_path: Optional[str] = None
    # 2026-05-28 slot-pool: per-expert GPU byte size for the fixed slot pool.
    # 0 = let archer auto-detect (max over registered expert nodes).  Set
    # explicitly per model only if auto-detect is undesirable.
    expert_byte_size: int = 0


@dataclass
class PolicyConfig:
    fetch_dispatch: str = "naive"
    placement: str = "naive_token_route"


@dataclass
class ExecutionConfig:
    # M7 overlap experiments: pre-trigger PCIe fetch for experts that will
    # land on this rank, so the fetch runs in parallel with the forward
    # NCCL all-to-all. No-op if archer doesn't expose enqueue_prefetch.
    enable_prefetch: bool = False


@dataclass
class InstrumentationConfig:
    enabled: bool = True
    trace_path: str = "/tmp/moe_ep_traces"


@dataclass
class Config:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    policies: PolicyConfig = field(default_factory=PolicyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    instrumentation: InstrumentationConfig = field(default_factory=InstrumentationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            parallel=ParallelConfig(**raw.get("parallel", {})),
            model=ModelConfig(**raw.get("model", {})),
            offload=OffloadConfig(**raw.get("offload", {})),
            policies=PolicyConfig(**raw.get("policies", {})),
            execution=ExecutionConfig(**raw.get("execution", {})),
            instrumentation=InstrumentationConfig(**raw.get("instrumentation", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parallel": self.parallel.__dict__,
            "model": self.model.__dict__,
            "offload": self.offload.__dict__,
            "policies": self.policies.__dict__,
            "execution": self.execution.__dict__,
            "instrumentation": self.instrumentation.__dict__,
        }

    def resolve_parallel(self, world_size: int) -> None:
        """Fill in ep_size=0 (auto -> world_size) and validate."""
        if self.parallel.ep_size == 0:
            self.parallel.ep_size = world_size
        if self.parallel.ep_size != world_size:
            raise ValueError(
                f"unified EP requires ep_size == world_size, got "
                f"ep_size={self.parallel.ep_size}, world_size={world_size}"
            )
