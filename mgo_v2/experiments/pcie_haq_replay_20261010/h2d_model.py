"""Fluid H2D completion model for 8 H100s, groups A=GPU0-3, B=GPU4-7.

Fitted (by subagent) to pcie_quota_r8_b16_20261010 all-8 anchors (max err <=4.2%)
and pcie_all8_concurrency_20261009 group-cap shape. Unbalanced quotas are
extrapolated; no calibration measured them.
"""
import numpy as np

GIB = 9 * 2**20 / 2**30
SOLO = 50.4
CAP_A = (0.0, 50.4, 100.0, 130.8, 133.4)
CAP_B = (0.0, 50.4, 100.6, 150.0, 187.5)
T_ALL = 321.0
T0 = np.array([.016] * 4 + [.040] * 4)


def _rates(act):
    a, b = int(act[:4].sum()), int(act[4:].sum())
    rA = min(SOLO, CAP_A[a] / a) if a else 0.0
    rB = min(SOLO, CAP_B[b] / b, (T_ALL - a * rA) / b) if b else 0.0
    return np.where(act, np.r_[[rA] * 4, [rB] * 4], 0.0)


def predict_rank_ms(q):
    q = np.asarray(q, float)
    left = q * GIB
    t = 0.0
    done = np.zeros(8)
    act = left > 0
    while act.any():
        r = _rates(act)
        dt = np.min(left[act] / r[act])
        left = np.where(act, left - r * dt, 0.0)
        t += dt
        fin = act & (left <= 1e-12)
        done[fin] = t
        act &= ~fin
    return np.where(q > 0, 1e3 * done + T0, 0.0)


if __name__ == '__main__':
    for q in ([2, 2, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 2, 2, 2, 2],
              [9] * 4 + [8] * 4, [8] * 4 + [9] * 4, [7] * 4 + [10] * 4):
        print(q, predict_rank_ms(q).round(3), predict_rank_ms(q).max().round(3))
