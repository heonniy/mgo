"""H2D completion-time model from the all-subset microbench (pcie_all8_concurrency_20261009).

Each rank r copies q_r experts of the payload size. While a set A of ranks still has bytes left, rank r
progresses at b_r(A), the measured per-rank GiB/s for exactly that active subset. The model returns
every rank's finish time (in units of payload / (GiB/s), i.e. expert copies per GiB/s); the makespan
is the max. This uses the burst microbench as-is (all ranks synchronized, back-to-back copies).
"""
import csv
from functools import lru_cache
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / 'pcie_all8_concurrency_20261009' / 'RANK_COMBINATIONS.csv'
W = 8


def load_rates(payload):
    rates = {}
    for row in csv.DictReader(open(BENCH)):
        if row['payload'] == payload:
            mask = sum(1 << int(g) for g in row['gpus'].split('-'))
            rates.setdefault(mask, [0.0] * W)[int(row['gpu'])] = float(row['rank_gib_per_s'])
    assert len(rates) == 255
    return rates


def finish_times(q, rates):
    left = [float(x) for x in q]
    t = 0.0
    done = [0.0] * W
    while True:
        mask = sum(1 << r for r in range(W) if left[r] > 1e-12)
        if not mask:
            return done
        b = rates[mask]
        dt = min(left[r] / b[r] for r in range(W) if mask >> r & 1)
        t += dt
        for r in range(W):
            if mask >> r & 1:
                left[r] -= dt * b[r]
                if left[r] <= 1e-9:
                    left[r] = 0.0
                    done[r] = t


def makespan(q, rates):
    return max(finish_times(q, rates))
