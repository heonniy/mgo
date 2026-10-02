"""Separate CPU-only exclusive component diagnostics; never installed in primary timing."""
from collections import defaultdict
from contextlib import contextmanager
import time
import types

from .admission import HungarianAdmission
from .controller_diagnostics import instrument, _plan, _hungarian, _choose
from .controller_optimized import IndexedCoverage

_opt_choose = instrument(IndexedCoverage.choose, {
    'slots =': 'eviction_candidates',
    'flat =': 'coverage_rank_and_victim',
    'self._sync_coverage(': 'coverage_sync_dispatch',
    'damage =': 'coverage_rank_and_victim',
}, 'eviction_candidates')


class ComponentDiagnostics:
    def __init__(self, controller):
        self.controller = controller
        self.records = []
        self.reset()
        controller._diag = self
        controller.plan_layer = types.MethodType(_plan, controller)
        eviction, admission, cache = controller.eviction, controller.admission, controller.cache
        eviction._diag = admission._diag = self
        optimized = isinstance(eviction, IndexedCoverage)
        choose = _opt_choose if optimized else _choose
        eviction.choose = types.MethodType(choose, eviction)
        self.wrap(eviction, 'choose', 'eviction_choose')
        self.wrap(eviction, 'candidate_slots' if optimized else 'candidates', None, count='candidate_visits')
        sync = eviction._sync_coverage
        def sync_coverage(value):
            if not optimized: self.counts['full_resident_set_materializations'] += 1
            with self.span('coverage_sync'):
                return sync(value)
        eviction._sync_coverage = sync_coverage
        self.wrap(cache, 'keys_on_rank', 'cache_keys', count='cache_key_items')
        resident = cache.resident_layer
        def resident_layer(layer):
            if not optimized or cache.layer_views[layer] is None:
                self.counts['layer_resident_materializations'] += 1
            with self.span('cache_resident_lookup'):
                return resident(layer)
        cache.resident_layer = resident_layer
        for name in ('place','evict','touch','next_tick'):
            self.wrap(cache, name, 'cache_mutation')
        self.wrap(cache, 'assert_consistent', 'cache_consistency_scan')
        if isinstance(admission, HungarianAdmission):
            admission.place = types.MethodType(_hungarian, admission)

    def reset(self):
        self.times = defaultdict(int);self.counts = defaultdict(int);self.stack = []

    @contextmanager
    def span(self, label):
        item = [time.perf_counter_ns(),0];self.stack.append(item)
        try:yield
        finally:
            elapsed = time.perf_counter_ns()-item[0];self.stack.pop()
            self.times[label] += elapsed-item[1]
            if self.stack:self.stack[-1][1] += elapsed

    def wrap(self, obj, name, label, count=None):
        original = getattr(obj,name)
        def wrapped(*args,**kwargs):
            self.counts[name+'_calls'] += 1
            if label:
                with self.span(label):result=original(*args,**kwargs)
            else:result=original(*args,**kwargs)
            if count:self.counts[count] += len(result)
            return result
        setattr(obj,name,wrapped)

    def plan_layer(self, routes):
        self.reset();start=time.perf_counter_ns()
        result=self.controller.plan_layer(routes)
        elapsed=time.perf_counter_ns()-start
        self.records.append(dict(event=len(self.records),layer=routes.layer,controller_ns=elapsed,
                                 times_ns=dict(self.times),counts=dict(self.counts),
                                 unattributed_ns=elapsed-sum(self.times.values())))
        return result
