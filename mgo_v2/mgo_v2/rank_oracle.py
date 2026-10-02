"""Opt-in exact diagnostic oracle. No production policy defaults are changed."""
from collections import Counter
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .admission import balanced_quotas
from .types import AdmissionResult


def demand(ctx):
    counts = Counter(e for token in ctx.effective_token_routes for e in token)
    base = [0] * ctx.world_size
    for e, r in ctx.preowned.items():
        base[r] += counts[e]
    return [counts[e] for e in sorted(ctx.incoming)], base


def solve_exact(d, base, quotas, time_limit=120):
    """Exact min-max MILP followed by a unique lexicographic assignment.

    Eight base-R rank digits at a time keep coefficients small. Fix every
    completed block, so ties cannot depend on solver thread scheduling.
    A timeout/nonoptimal status invalidates the event; no fallback exists.
    """
    started = time.perf_counter()
    d, base, quotas = list(d), list(base), list(quotas)
    n, w = len(d), len(base)
    if not w or len(quotas) != w or sum(quotas) != n:
        raise ValueError('invalid quota dimensions')
    if any(int(x) != x or x < 0 for x in d + base + quotas):
        raise ValueError('loads/quotas must be nonnegative integers')
    if not n:
        return [], dict(z=max(base), loads=base, solver_seconds=0., solves=0, status='optimal')
    size = n*w+1
    a = lil_matrix((n+2*w, size), dtype=float)
    lower = np.full(n+2*w, -np.inf)
    upper = np.zeros(n+2*w)
    for i in range(n):
        a[i, i*w:(i+1)*w] = 1
        lower[i] = upper[i] = 1
    for r in range(w):
        a[n+r, r:n*w:w] = 1
        lower[n+r] = upper[n+r] = quotas[r]
        a[n+w+r, r:n*w:w] = d
        a[n+w+r, -1] = -1
        upper[n+w+r] = -base[r]
    lo = np.zeros(size)
    hi = np.ones(size)
    hi[-1] = max(base) + sum(d)
    constraints = LinearConstraint(a.tocsc(), lower, upper)
    calls = 0
    certificate = {}
    def run(c):
        nonlocal calls
        calls += 1
        # HiGHS options passed through SciPy; one thread bounds CPU use.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected')
            result = milp(c, integrality=np.ones(size), bounds=Bounds(lo, hi),
                          constraints=constraints, options=dict(mip_rel_gap=0.,
                          time_limit=time_limit, threads=1, random_seed=42))
        if result.status != 0 or not result.success:
            raise RuntimeError(f'exact oracle failed (solve {calls}): {result.message}')
        certificate.update(primal_objective=float(result.fun),
                           dual_bound=float(result.mip_dual_bound),
                           mip_gap=float(result.mip_gap),status=int(result.status))
        x = np.rint(result.x)
        if np.max(np.abs(x-result.x)) > 1e-4:
            raise RuntimeError('nonintegral oracle solution')
        values = a @ x
        if np.any(values < lower-1e-6) or np.any(values > upper+1e-6):
            raise RuntimeError('infeasible rounded oracle solution')
        return x
    c = np.zeros(size); c[-1] = 1
    first = run(c); z = int(first[-1])
    primary_certificate = dict(certificate)
    lo[-1] = hi[-1] = z
    for begin in range(0, n, 8):
        end = min(n, begin+8)
        c = np.zeros(size)
        for i in range(begin, end):
            c[i*w:(i+1)*w] = np.arange(w) * w**(end-i-1)
        x = run(c)
        lo[begin*w:end*w] = hi[begin*w:end*w] = x[begin*w:end*w]
    ranks = x[:-1].reshape(n, w).argmax(axis=1).tolist()
    loads = [base[r]+sum(v for v, owner in zip(d, ranks) if owner == r) for r in range(w)]
    if max(loads) != z or [ranks.count(r) for r in range(w)] != quotas:
        raise RuntimeError('oracle objective/quota verification failed')
    return ranks, dict(z=z, loads=loads, solver_seconds=time.perf_counter()-started,
                       solves=calls, status='optimal', primary_certificate=primary_certificate)


class ExactRankDemandOracle:
    name = 'exact_rank_demand_oracle'
    def __init__(self):
        self.records = []
    def place(self, ctx):
        incoming = sorted(ctx.incoming)
        quotas = balanced_quotas(len(incoming), ctx.world_size)
        d, base = demand(ctx)
        ranks, record = solve_exact(d, base, quotas)
        assignment = dict(zip(incoming, ranks))
        self.records.append(dict(layer=ctx.layer, incoming=incoming, demand=d,
                                 base=base, quotas=quotas, assignment=assignment, **record))
        return AdmissionResult(assignment, quotas, self.name)


class FrozenRankDemandOracle:
    name = 'exact_rank_demand_oracle'
    def __init__(self, records):
        self.records, self.cursor = records, 0
    def place(self, ctx):
        if self.cursor >= len(self.records):
            raise RuntimeError('frozen oracle trace exhausted')
        record = self.records[self.cursor]; self.cursor += 1
        incoming = sorted(ctx.incoming)
        quotas = balanced_quotas(len(incoming), ctx.world_size)
        d, base = demand(ctx)
        actual = (ctx.layer, incoming, d, base, quotas)
        expected = tuple(record[k] for k in ('layer', 'incoming', 'demand', 'base', 'quotas'))
        if actual != expected:
            raise RuntimeError(f'frozen oracle demand divergence at event {self.cursor-1}')
        assignment = {int(e): int(r) for e, r in record['assignment'].items()}
        if sorted(assignment) != incoming or any(r < 0 or r >= ctx.world_size for r in assignment.values()):
            raise RuntimeError('invalid frozen assignment')
        if [list(assignment.values()).count(r) for r in range(ctx.world_size)] != quotas:
            raise RuntimeError('frozen quota violation')
        return AdmissionResult(assignment, quotas, self.name)
