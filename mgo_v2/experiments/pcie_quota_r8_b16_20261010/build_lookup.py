"""Build balanced miss-count quotas from all-subset short-burst H2D receipts."""

import hashlib
import itertools
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/pcie_quota_r8_b16_20261010')
COPIES = (1, 2, 4, 8, 32)
WORLD = 8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_calibration():
    durations = {}
    sources = {}
    all_groups = {tuple(group) for n in range(1, WORLD + 1)
                  for group in itertools.combinations(range(WORLD), n)}
    for copies in COPIES:
        path = RAW / f'RAW_c{copies}.json'
        rows = json.loads(path.read_text())
        assert len(rows) == (510 if copies == 1 else 2)
        refine_path = RAW / f'REFINE_c{copies}.json'
        refined = json.loads(refine_path.read_text())
        sources[str(copies)] = dict(path=str(path), sha256=sha(path),
                                    refine_path=str(refine_path),
                                    refine_sha256=sha(refine_path))
        samples = {}
        for row in rows:
            group = tuple(row['gpus'])
            assert group in all_groups and row['bytes'] == 9 * 1024 * 1024
            assert row['copies'] == copies and row['repeat'] in (1, 2)
            for rank in group:
                samples.setdefault((group, rank), []).append(
                    float(row['rank_results'][str(rank)]['event_seconds']))
        refined_samples = {}
        for row in refined:
            group = tuple(row['gpus'])
            assert group in all_groups and row['copies'] == copies
            if row['start_skew_ms'] > .2:
                continue  # quality gate on launch timing, independent of measured bandwidth
            for rank in group:
                refined_samples.setdefault((group, rank), []).append(
                    float(row['rank_results'][str(rank)]['event_seconds']))
        assert all(len(values) >= 12 for values in refined_samples.values())
        for key, values in refined_samples.items():
            samples[key] = values
        assert set(group for group, _ in samples) == (all_groups if copies == 1 else {tuple(range(WORLD))})
        assert all(len(value) in (2, 12, 13, 14, 15, 16) for value in samples.values())
        for key, values in samples.items():
            durations.setdefault(key, {})[copies] = statistics.median(values)
    # A longer equal-copy burst cannot physically complete sooner than a
    # shorter one. Preserve raw values; monotonicize only the lookup model.
    for series in durations.values():
        running = 0.0
        for copies in sorted(series):
            running = max(running, series[copies])
            series[copies] = running
    return durations, sources


def duration(series, copies):
    if copies == 0:
        return 0.0
    if copies in series:
        return series[copies]
    for lo, hi in zip(COPIES, COPIES[1:]):
        if lo < copies < hi:
            return series[lo] + (series[hi] - series[lo]) * (copies - lo) / (hi - lo)
    assert copies > 32
    return series[32] + (copies - 32) * max(0.0, (series[32] - series[8]) / 24)


def predicted(quota, calibration):
    group = tuple(rank for rank, count in enumerate(quota) if count)
    if not group:
        return 0.0, 0.0
    values = [duration(calibration[(group if count == 1 else tuple(range(WORLD)), rank)], count)
              for rank, count in enumerate(quota) if count]
    return max(values), sum(values)


def quotas(n):
    base, extra = divmod(n, WORLD)
    for ranks in itertools.combinations(range(WORLD), extra):
        selected = set(ranks)
        yield [base + int(rank in selected) for rank in range(WORLD)]


def main():
    calibration, sources = read_calibration()
    fast_order = sorted(range(WORLD), key=lambda rank:
                        (calibration[(tuple(range(WORLD)), rank)][8], rank))
    tables = {'NEAR_PCIE': [], 'NEAR_FAST': []}
    cases = []
    for n in range(129):
        options = list(quotas(n))
        best = min(options, key=lambda quota: (*predicted(quota, calibration), quota))
        base, rem = divmod(n, WORLD)
        fast = [base + int(rank in fast_order[:rem]) for rank in range(WORLD)]
        assert fast in options
        tables['NEAR_PCIE'].append(best)
        tables['NEAR_FAST'].append(fast)
        best_time, _ = predicted(best, calibration)
        fast_time, _ = predicted(fast, calibration)
        assert best_time <= fast_time + 1e-9
        cases.append(dict(misses=n, pcie_quota=best, fast_quota=fast,
                          predicted_pcie_ms=1000 * best_time,
                          predicted_fast_ms=1000 * fast_time))
    result = dict(status='PASS', physical_gpus=list(range(WORLD)),
                  payload_bytes=9 * 1024 * 1024,
                  calibration_sources=sources,
                  prediction='minimum max-rank CUDA-event H2D duration; short-burst subset interpolation',
                  fastest_full8_order=fast_order,
                  quota_lut=tables, cases=cases)
    (HERE / 'LOOKUP.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', fastest_full8_order=fast_order,
                          n12=cases[12], changed=sum(a != b for a, b in
                          zip(tables['NEAR_PCIE'], tables['NEAR_FAST'])))))


if __name__ == '__main__':
    main()
