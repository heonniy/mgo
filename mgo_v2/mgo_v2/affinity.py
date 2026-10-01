from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AffinityTables:
    same_layer: np.ndarray | None = None  # [L,E,E]
    path: np.ndarray | None = None        # [L-1,E,E]

    @classmethod
    def load_npz(cls, path: str) -> "AffinityTables":
        data = np.load(path)
        return cls(
            same_layer=data["same_layer"] if "same_layer" in data else None,
            path=data["path"] if "path" in data else None,
        )

    def same_score(self, layer: int, expert: int, residents: list[int]) -> float:
        if self.same_layer is None or not residents:
            return 0.0
        return float(np.sum(self.same_layer[layer, expert, residents]))

    def path_score(
        self,
        layer: int,
        expert: int,
        prev_residents: list[int],
        next_residents: list[int],
    ) -> float:
        if self.path is None:
            return 0.0
        score = 0.0
        sides = 0
        if layer > 0 and prev_residents:
            score += float(np.sum(self.path[layer - 1, prev_residents, expert]))
            sides += 1
        if layer < self.path.shape[0] and next_residents:
            score += float(np.sum(self.path[layer, expert, next_residents]))
            sides += 1
        return score / max(1, sides)
