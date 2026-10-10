"""Measure the compiled Near assignment cost with and without a quota row."""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts'))
from la_placement import load_locality_near_assignment


@njit
def batch_default(demand, misses, owner, iterations):
    total = 0
    for _ in range(iterations):
        total += load_locality_near_assignment(demand, misses, owner, 0, 200).sum()
    return total


@njit
def batch_lookup(demand, misses, owner, table, iterations):
    total = 0
    for _ in range(iterations):
        total += load_locality_near_assignment(
            demand, misses, owner, 0, 200, table[len(misses)]).sum()
    return total


def main():
    table = np.asarray(json.loads((HERE / 'LOOKUP.json').read_text())
                       ['quota_lut']['NEAR_PCIE'], dtype=np.int64)
    demand = np.random.default_rng(42).integers(0, 256, (128, 8), dtype=np.int64)
    owner = np.zeros(48 * 128, dtype=np.int16)
    rows = []
    for n in (12, 64, 80):
        misses = np.arange(n, dtype=np.int64)
        batch_default(demand, misses, owner, 1)
        batch_lookup(demand, misses, owner, table, 1)
        measures = {}
        for name, function in (('original_near', batch_default),
                               ('lookup_near', batch_lookup)):
            values = []
            for _ in range(7):
                start = time.perf_counter_ns()
                if name == 'original_near':
                    function(demand, misses, owner, 200)
                else:
                    function(demand, misses, owner, table, 200)
                values.append((time.perf_counter_ns() - start) / 200 / 1000)
            measures[name] = dict(median_us=statistics.median(values),
                                  minimum_us=min(values), maximum_us=max(values))
        rows.append(dict(misses=n, **measures))
    (HERE / 'LOOKUP_OVERHEAD.json').write_text(json.dumps(
        dict(status='PASS', compiled_numba=True, iterations_per_sample=200,
             groups=7, rows=rows), indent=2) + '\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
