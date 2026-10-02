"""Opt-in, CPU-only controller diagnostics; the default controller is untouched.

Instrumented functions are built from the live validated source, inserting only
context-manager spans around top-level statements. No policy statement is
rewritten. Nested spans report exclusive nanoseconds. Trace serialization and
trajectory accounting happen outside plan timers.
"""
from __future__ import annotations
import ast
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
import pickle
import textwrap
import time
import types

import numpy as np
from .controller import GlobalExpertController
from .admission import HungarianAdmission
from .eviction import DiversityEviction


def canonical(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return [[canonical(k), canonical(v)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))]
    if isinstance(value, (set, frozenset)):
        return [canonical(v) for v in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), separators=(',', ':')).encode()).hexdigest()


def instrument(function, boundaries, default):
    """Wrap consecutive original statements; fail if a boundary disappeared."""
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    node = tree.body[0]
    groups = []
    label = default
    found = set()
    for statement in node.body:
        line = source.splitlines()[statement.lineno - 1].strip()
        for prefix, name in boundaries.items():
            if line.startswith(prefix):
                label = name
                found.add(prefix)
                break
        if not groups or groups[-1][0] != label:
            groups.append((label, []))
        groups[-1][1].append(statement)
    if found != set(boundaries):
        raise RuntimeError(f'diagnostic boundary drift: {set(boundaries) - found}')
    node.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                             attr='_diag', ctx=ast.Load()), attr='span', ctx=ast.Load()),
        args=[ast.Constant(label)], keywords=[]))], body=body)
        for label, body in groups]
    # Removing the inserted span nodes must recover precisely the original AST.
    stripped = ast.parse(source)
    stripped.body[0].body = [s for wrapper in node.body for s in wrapper.body]
    assert ast.dump(stripped) == ast.dump(ast.parse(source))
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), function.__code__.co_filename + ':diagnostic', 'exec'), namespace)
    result = namespace[function.__name__]
    result.original_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return result


_plan = instrument(GlobalExpertController.plan_layer, {
    'if routes.full_router_probs': 'history',
    'decision =': 'substitution',
    'effective =': 'effective_route_merge',
    'preowned = {': 'admission_prep',
    'admission =': 'admission_policy',
    'local_exec =': 'cache_mutation_and_bookkeeping',
}, 'cache_mutation_and_bookkeeping')
_hungarian = instrument(HungarianAdmission.place, {
    'rank_cost =': 'admission_cost_build',
    'rows, cols =': 'admission_assignment',
    'out =': 'admission_result',
}, 'admission_prep')
_choose = instrument(DiversityEviction.choose, {
    'cand =': 'eviction_candidates',
    'gate =': 'coverage_rank_and_victim',
    'self._sync_coverage(': 'coverage_sync_dispatch',
    'damage =': 'coverage_rank_and_victim',
}, 'eviction_candidates')


class ControllerDiagnostics:
    def __init__(self, controller, event_path=None, route_path=None):
        self.controller = controller
        self.event_file = open(event_path, 'w') if event_path else None
        self.route_file = open(route_path, 'wb') if route_path else None
        self.seen = set()
        self.event_index = 0
        self.records = [] if event_path is None else None
        self.reset()
        controller._diag = self
        # Only these two frozen policies and Coverage are in the experiment.
        assert controller.config.admission in ('random', 'hungarian_current')
        assert isinstance(controller.eviction, DiversityEviction)
        admission, eviction = controller.admission, controller.eviction
        admission._diag = eviction._diag = self
        if isinstance(admission, HungarianAdmission):
            admission.place = types.MethodType(_hungarian, admission)
        self._wrap(eviction, 'choose', 'eviction_choose', replacement=types.MethodType(_choose, eviction))
        self._wrap(eviction, 'candidates', None, result_count='eviction_candidate_count')
        sync = eviction._sync_coverage
        def sync_coverage(cache):
            with self.span('diagnostic_accounting'):
                old = eviction._coverage_residents if eviction._coverage_cache is cache else set()
                added, removed = set(cache.owner) - old, old - set(cache.owner)
                self.counts['coverage_resident_additions'] += len(added)
                self.counts['coverage_resident_removals'] += len(removed)
                self.counts['coverage_changed_layers'] += len({k[0] for k in added | removed})
            with self.span('coverage_sync'):
                return sync(cache)
        eviction._sync_coverage = sync_coverage
        self._wrap(controller.cache, 'keys_on_rank', 'cache_keys_on_rank', result_count='cache_keys_materialized')
        controller.plan_layer = self.plan_layer

    def _wrap(self, obj, name, label, result_count=None, replacement=None):
        original = replacement or getattr(obj, name)
        def wrapped(*args, **kwargs):
            self.counts[name + '_calls'] += 1
            if label:
                with self.span(label):
                    result = original(*args, **kwargs)
            else:
                result = original(*args, **kwargs)
            if result_count:
                self.counts[result_count] += len(result)
            return result
        setattr(obj, name, wrapped)

    def reset(self):
        self.times = defaultdict(int)
        self.calls = defaultdict(int)
        self.counts = defaultdict(int)
        self.stack = []

    @contextmanager
    def span(self, label):
        item = [time.perf_counter_ns(), 0]
        self.stack.append(item)
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - item[0]
            self.stack.pop()
            self.times[label] += elapsed - item[1]
            self.calls[label] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def plan_layer(self, routes):
        self.reset()
        self.before_free = [r.capacity - len(r) for r in self.controller.cache.ranks]
        start = time.perf_counter_ns()
        plan = _plan(self.controller, routes)
        total = time.perf_counter_ns() - start
        self.last = self.record(routes, plan, total)
        if self.route_file:
            pickle.dump(routes, self.route_file, protocol=5)
        if self.event_file:
            self.event_file.write(json.dumps(self.last, separators=(',', ':')) + '\n')
        elif self.records is not None:
            self.records.append(self.last)
        self.event_index += 1
        return plan

    def record(self, routes, plan, total):
        world = self.controller.config.world_size
        loads = np.zeros(world, dtype=np.int64)
        expert_rows = np.zeros(world, dtype=np.int64)
        remote = local = remote_routes = 0
        for origin, effective in zip(routes.origin_ranks, plan.effective_token_routes):
            owners = [plan.owner_by_expert[e] for e in effective]
            for r in owners:
                expert_rows[r] += 1
                remote_routes += r != int(origin)
            for r in set(owners):
                loads[r] += 1
                remote += r != int(origin)
                local += r == int(origin)
        admissions, evictions = [], []
        for rank, execution in plan.local_exec.items():
            for layer, expert, slot, vl, ve, order in execution.miss_ops:
                admissions.append([layer, expert, rank, slot])
                if vl >= 0:
                    evictions.append([vl, ve, rank, slot])
        incoming = {(a[0], a[1]) for a in admissions}
        reloads = len(incoming & self.seen)
        self.seen.update(incoming)
        self.counts['incoming_experts'] = len(incoming)
        self.counts['hungarian_rank_cost_elements'] = len(incoming) * world if self.controller.config.admission == 'hungarian_current' else 0
        self.counts['hungarian_assignment_elements'] = len(incoming) ** 2 if self.controller.config.admission == 'hungarian_current' else 0
        self.counts['raw_token_rows'] = len(routes.origin_ranks)
        self.counts['effective_expert_routes'] = int(expert_rows.sum())
        index = self.event_index
        return dict(event=index, step=index // self.controller.config.num_layers, layer=routes.layer,
            phase='prefill' if index < self.controller.config.num_layers else 'decode',
            controller_ns=total, times_ns=dict(self.times), calls=dict(self.calls), counts=dict(self.counts),
            timer_unattributed_ns=total - sum(self.times.values()),
            admissions=admissions, evictions=evictions, reloads=reloads,
            hit=len(plan.substitution.exact_hits), subhit=len(plan.substitution.source_to_target), miss=len(incoming),
            raw_sources=sorted(set(routes.selected_experts.ravel().tolist())),
            substitutes=sorted(plan.substitution.source_to_target.items()),
            owners=sorted(plan.owner_by_expert.items()), quotas=plan.admission.quotas,
            free_before=self.before_free, free_after=[r.capacity - len(r) for r in self.controller.cache.ranks],
            remote_pairs=int(remote), local_pairs=int(local), remote_expert_routes=int(remote_routes),
            rank_tokens=loads.tolist(), rank_expert_rows=expert_rows.tolist(),
            rank_token_cv=float(loads.std() / max(loads.mean(), 1e-30)),
            rank_token_max_mean=float(loads.max() / max(loads.mean(), 1e-30)),
            plan_sha256=digest(plan), cache_sha256=digest(self.controller.cache.owner))

    def close(self):
        for file in (self.event_file, self.route_file):
            if file:
                file.close()


def enable_diagnostics(controller, event_path=None, route_path=None):
    return ControllerDiagnostics(controller, event_path, route_path)
