from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RuntimeConfig:
    num_layers: int = 48
    num_experts: int = 128
    top_k: int = 8

    gate_protect_threshold: float = 0.20
    similarity_threshold: float = 0.65

    global_cache_ratio: float = 0.30
    gate_window: int = 128
    coverage_k: int = 1
    coverage_lambda: float = 2.0

    world_size: int = 4
    admission: Literal[
        "random",
        "greedy_current",
        "greedy_path",
        "hungarian_current",
        "hungarian_same",
        "hungarian_same_path",
        "hungarian_swap",
    ] = "hungarian_same_path"
    eviction: Literal["lru", "gate", "coverage"] = "coverage"

    same_layer_alpha: float = 1.0
    path_eta: float = 0.5
    seed: int = 42

    def total_experts(self) -> int:
        return self.num_layers * self.num_experts

    def global_slots(self) -> int:
        return int(self.total_experts() * self.global_cache_ratio)

    def per_rank_slots(self) -> list[int]:
        total = self.global_slots()
        base, rem = divmod(total, self.world_size)
        return [base + (1 if r < rem else 0) for r in range(self.world_size)]
