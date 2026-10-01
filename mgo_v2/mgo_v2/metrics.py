from __future__ import annotations

from dataclasses import dataclass, asdict

from .types import SubstitutionResult


@dataclass
class ExpertMetrics:
    hit: int = 0
    subhit: int = 0
    miss: int = 0
    fetches: int = 0
    remote_token_rank_pairs: int = 0

    def update_substitution(self, result: SubstitutionResult) -> None:
        self.hit += len(result.exact_hits)
        self.subhit += len(result.source_to_target)
        self.miss += len(result.residual_exact_misses)
        self.fetches += len(result.residual_exact_misses)

    def rates(self) -> dict[str, float]:
        total = self.hit + self.subhit + self.miss
        if total == 0:
            return {"expert_hit": 0.0, "expert_subhit": 0.0, "expert_miss": 0.0}
        return {
            "expert_hit": self.hit / total,
            "expert_subhit": self.subhit / total,
            "expert_miss": self.miss / total,
        }

    def to_dict(self):
        out = asdict(self)
        out.update(self.rates())
        return out
