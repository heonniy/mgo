from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .affinity import AffinityTables
from .cache import GlobalCacheState
from .types import AdmissionResult


def balanced_quotas(n: int, world_size: int) -> list[int]:
    base, rem = divmod(n, world_size)
    return [base + (1 if r < rem else 0) for r in range(world_size)]


@dataclass
class AdmissionContext:
    layer: int
    incoming: list[int]
    origin_ranks: np.ndarray
    effective_token_routes: list[dict[int, float]]
    preowned: dict[int, int]  # execution expert -> rank before admissions
    cache: GlobalCacheState
    world_size: int
    affinity: AffinityTables | None = None


def _token_sources(ctx: AdmissionContext) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {e: [] for e in ctx.incoming}
    incoming = set(ctx.incoming)
    for t, routes in enumerate(ctx.effective_token_routes):
        for e in routes:
            if e in incoming:
                out[e].append(t)
    return out


def _base_destinations(ctx: AdmissionContext) -> list[set[int]]:
    dests = [set() for _ in ctx.effective_token_routes]
    for t, routes in enumerate(ctx.effective_token_routes):
        for e in routes:
            if e in ctx.preowned:
                dests[t].add(ctx.preowned[e])
    return dests


def _incremental_cost(
    ctx: AdmissionContext,
    expert: int,
    rank: int,
    token_sources: dict[int, list[int]],
    base_dests: list[set[int]],
) -> float:
    cost = 0
    for t in token_sources.get(expert, []):
        if rank == int(ctx.origin_ranks[t]):
            continue
        if rank in base_dests[t]:
            continue
        cost += 1
    return float(cost)


def exact_remote_pairs(
    ctx: AdmissionContext, assignment: dict[int, int]
) -> int:
    owners = dict(ctx.preowned)
    owners.update(assignment)
    total = 0
    for t, routes in enumerate(ctx.effective_token_routes):
        destinations = {owners[e] for e in routes}
        origin = int(ctx.origin_ranks[t])
        total += sum(1 for r in destinations if r != origin)
    return total


class AdmissionPolicy:
    name = "base"

    def place(self, ctx: AdmissionContext) -> AdmissionResult:
        raise NotImplementedError


class BalancedRandomAdmission(AdmissionPolicy):
    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def place(self, ctx):
        quotas = balanced_quotas(len(ctx.incoming), ctx.world_size)
        remaining = quotas.copy()
        order = sorted(ctx.incoming)
        self.rng.shuffle(order)
        out = {}
        for e in order:
            legal = [r for r in range(ctx.world_size) if remaining[r] > 0]
            r = self.rng.choice(legal)
            out[e] = r
            remaining[r] -= 1
        return AdmissionResult(out, quotas, self.name)


class GreedyCurrentAdmission(AdmissionPolicy):
    name = "greedy_current"

    def place(self, ctx):
        quotas = balanced_quotas(len(ctx.incoming), ctx.world_size)
        remaining = quotas.copy()
        token_sources = _token_sources(ctx)
        base = _base_destinations(ctx)
        out = {}
        for e in sorted(ctx.incoming):
            legal = [r for r in range(ctx.world_size) if remaining[r] > 0]
            r = min(
                legal,
                key=lambda x: (_incremental_cost(ctx, e, x, token_sources, base), x),
            )
            out[e] = r
            remaining[r] -= 1
            for t in token_sources[e]:
                base[t].add(r)
        return AdmissionResult(out, quotas, self.name)


class GreedyPathAdmission(GreedyCurrentAdmission):
    name = "greedy_path"

    def __init__(self, path_eta: float = 0.5):
        self.path_eta = path_eta

    def place(self, ctx):
        quotas = balanced_quotas(len(ctx.incoming), ctx.world_size)
        remaining = quotas.copy()
        token_sources = _token_sources(ctx)
        base = _base_destinations(ctx)
        out = {}
        for e in sorted(ctx.incoming):
            legal = [r for r in range(ctx.world_size) if remaining[r] > 0]
            def cost(r):
                comm = _incremental_cost(ctx, e, r, token_sources, base)
                path = 0.0
                if ctx.affinity is not None:
                    prev = [x[1] for x in ctx.cache.keys_on_rank(r, ctx.layer - 1)]
                    nxt = [x[1] for x in ctx.cache.keys_on_rank(r, ctx.layer + 1)]
                    path = ctx.affinity.path_score(ctx.layer, e, prev, nxt)
                return (comm - self.path_eta * path, r)
            r = min(legal, key=cost)
            out[e] = r
            remaining[r] -= 1
            for t in token_sources[e]:
                base[t].add(r)
        return AdmissionResult(out, quotas, self.name)


class HungarianAdmission(AdmissionPolicy):
    def __init__(
        self,
        use_same: bool = False,
        use_path: bool = False,
        same_alpha: float = 1.0,
        path_eta: float = 0.5,
        name: str | None = None,
    ):
        self.use_same = use_same
        self.use_path = use_path
        self.same_alpha = same_alpha
        self.path_eta = path_eta
        self.name = name or (
            "hungarian_same_path" if use_same and use_path
            else "hungarian_same" if use_same
            else "hungarian_current"
        )

    def _cost(self, ctx, e, r, token_sources, base):
        value = _incremental_cost(ctx, e, r, token_sources, base)
        if ctx.affinity is None:
            return value
        if self.use_same:
            residents = [x[1] for x in ctx.cache.keys_on_rank(r, ctx.layer)]
            value -= self.same_alpha * ctx.affinity.same_score(
                ctx.layer, e, residents
            )
        if self.use_path:
            prev = [x[1] for x in ctx.cache.keys_on_rank(r, ctx.layer - 1)]
            nxt = [x[1] for x in ctx.cache.keys_on_rank(r, ctx.layer + 1)]
            value -= self.path_eta * ctx.affinity.path_score(
                ctx.layer, e, prev, nxt
            )
        return value

    def place(self, ctx):
        incoming = sorted(ctx.incoming)
        quotas = balanced_quotas(len(incoming), ctx.world_size)
        slots = [r for r, q in enumerate(quotas) for _ in range(q)]
        if not incoming:
            return AdmissionResult({}, quotas, self.name)

        token_sources = _token_sources(ctx)
        base = _base_destinations(ctx)
        # Quota slots on the same rank have identical costs. Evaluate each
        # expert/rank pair once, then expand to the assignment matrix.
        rank_cost = np.zeros((len(incoming), ctx.world_size), dtype=np.float64)
        for i, e in enumerate(incoming):
            for r in range(ctx.world_size):
                rank_cost[i, r] = self._cost(ctx, e, r, token_sources, base)
        cost = rank_cost[:, slots]

        rows, cols = linear_sum_assignment(cost)
        out = {incoming[i]: slots[j] for i, j in zip(rows.tolist(), cols.tolist())}
        return AdmissionResult(out, quotas, self.name)


class SwapRefinedAdmission(AdmissionPolicy):
    name = "hungarian_swap"

    def __init__(self, seed_policy: HungarianAdmission, max_passes: int = 2):
        self.seed_policy = seed_policy
        self.max_passes = max_passes

    def place(self, ctx):
        base_result = self.seed_policy.place(ctx)
        assign = dict(base_result.expert_to_rank)
        experts = sorted(assign)
        owners = dict(ctx.preowned)
        owners.update(assign)
        # A swap changes only two destination counts on tokens that route to
        # exactly one of its experts. Keep integer counts to evaluate the same
        # objective without rebuilding every token's destination set per trial.
        counts = np.zeros((len(ctx.effective_token_routes), ctx.world_size), dtype=np.int32)
        for t, routes in enumerate(ctx.effective_token_routes):
            for expert in routes:
                counts[t, owners[expert]] += 1
        presence = np.zeros((len(experts), len(ctx.effective_token_routes)), dtype=bool)
        sources = _token_sources(ctx)
        for i, expert in enumerate(experts):
            presence[i, sources[expert]] = True

        for _ in range(self.max_passes):
            improved = False
            for i, a in enumerate(experts):
                for j in range(i + 1, len(experts)):
                    b = experts[j]
                    ra, rb = assign[a], assign[b]
                    if ra == rb:
                        continue
                    tokens = np.flatnonzero(presence[i] != presence[j])
                    delta = np.where(presence[i, tokens], 1, -1)
                    ca, cb = counts[tokens, ra], counts[tokens, rb]
                    remote_a = ctx.origin_ranks[tokens] != ra
                    remote_b = ctx.origin_ranks[tokens] != rb
                    before = np.count_nonzero((ca > 0) & remote_a) + np.count_nonzero((cb > 0) & remote_b)
                    after = np.count_nonzero((ca - delta > 0) & remote_a) + np.count_nonzero((cb + delta > 0) & remote_b)
                    if after < before:
                        assign[a], assign[b] = rb, ra
                        counts[tokens, ra] -= delta
                        counts[tokens, rb] += delta
                        improved = True
            if not improved:
                break

        return AdmissionResult(assign, base_result.quotas, self.name)
