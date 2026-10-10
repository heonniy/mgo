"""CPU-only checks for compiled PCIe quota lookup and Near assignment."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import mgo_v2  # initialize the package before importing its script adapter
from env_offload_policy import Policy
from la_placement import load_locality_near_assignment


def table(extra_order):
    rows = []
    for n in range(129):
        base, rem = divmod(n, 8)
        selected = set(extra_order[:rem])
        rows.append([base + int(rank in selected) for rank in range(8)])
    return np.asarray(rows, dtype=np.int64)


def test_compiled_near_uses_exact_quota_row():
    quota = table([7, 6, 5, 4, 3, 2, 1, 0])
    demand = np.arange(128 * 8, dtype=np.int64).reshape(128, 8) % 17
    owner = np.zeros(48 * 128, dtype=np.int16)
    for n in (0, 3, 8, 12, 31, 128):
        misses = np.arange(n, dtype=np.int64)
        assignment = load_locality_near_assignment(
            demand, misses, owner, 0, 200, quota[n])
        assert np.array_equal(np.bincount(assignment, minlength=8), quota[n])


def test_policy_fetches_follow_lookup_not_rank_number():
    quota = table([7, 6, 5, 4, 3, 2, 1, 0])
    p = Policy([16] * 8, np.zeros((48, 128, 128), np.float32),
               False, 11, quota_lut=quota)
    selected = np.repeat(np.arange(12, dtype=np.int16)[:, None], 2, axis=1)
    weights = np.full((12, 2), .5, dtype=np.float32)
    origins = np.arange(12, dtype=np.int64) % 8
    result = p.apply(0, selected, weights, origins,
                     np.zeros(128, np.float32), np.zeros((128, 8), np.int32))
    counts = np.bincount([row[0] for row in result[5]], minlength=8)
    assert np.array_equal(counts, quota[12])
    assert np.array_equal(counts, [1, 1, 1, 1, 2, 2, 2, 2])


def test_original_near_matches_identical_balanced_lookup():
    quota = table(list(range(8)))
    similarity = np.zeros((48, 128, 128), np.float32)
    policies = [Policy([16] * 8, similarity, False, 7),
                Policy([16] * 8, similarity, False, 11, quota_lut=quota)]
    selected = np.repeat(np.arange(12, dtype=np.int16)[:, None], 2, axis=1)
    weights = np.full((12, 2), .5, dtype=np.float32)
    origins = np.arange(12, dtype=np.int64) % 8
    rows = [policy.apply(0, selected, weights, origins,
                         np.zeros(128, np.float32), np.zeros((128, 8), np.int32))
            for policy in policies]
    for field in range(5):
        assert np.array_equal(rows[0][field], rows[1][field])
    assert rows[0][5] == rows[1][5]
    assert np.array_equal(rows[0][6], rows[1][6])


def test_split_policy_follows_unbalanced_lookup():
    quota = table([7, 6, 5, 4, 3, 2, 1, 0])
    quota[12] = [1, 1, 1, 0, 3, 2, 2, 2]
    p = Policy([16] * 8, np.zeros((48, 128, 128), np.float32),
               False, 20, quota_lut=quota)
    selected = np.repeat(np.arange(12, dtype=np.int16)[:, None], 2, axis=1)
    weights = np.full((12, 2), .5, dtype=np.float32)
    origins = np.arange(12, dtype=np.int64) % 8
    result = p.apply(0, selected, weights, origins,
                     np.zeros(128, np.float32), np.zeros((128, 8), np.int32))
    counts = np.bincount([row[0] for row in result[5]], minlength=8)
    assert np.array_equal(counts, [1, 1, 1, 0, 3, 2, 2, 2])


def test_split_policy_matches_fast_on_balanced_rows():
    quota = table([7, 6, 5, 4, 3, 2, 1, 0])
    similarity = np.zeros((48, 128, 128), np.float32)
    policies = [Policy([16] * 8, similarity, False, 12, quota_lut=quota),
                Policy([16] * 8, similarity, False, 20, quota_lut=quota)]
    selected = np.repeat(np.arange(12, dtype=np.int16)[:, None], 2, axis=1)
    weights = np.full((12, 2), .5, dtype=np.float32)
    origins = np.arange(12, dtype=np.int64) % 8
    rows = [policy.apply(0, selected, weights, origins,
                         np.zeros(128, np.float32), np.zeros((128, 8), np.int32))
            for policy in policies]
    for field in range(5):
        assert np.array_equal(rows[0][field], rows[1][field])


def test_split_lookup_rows_are_valid():
    import json
    data = json.loads((Path(__file__).resolve().parents[1] /
                       'experiments/pcie_haq_replay_20261010/SPLIT_LOOKUP.json').read_text())
    lut = np.asarray(data['quota_lut']['NEAR_SPLIT'])
    assert data['status'] == 'PASS' and lut.shape == (129, 8)
    assert all(lut[n].sum() == n and lut[n].min() >= 0 for n in range(129))


def test_haq_fast_quota_prefers_fast_group_under_cap():
    from haq_placement import hit_aware_quota_rank_cost, H2D_US_BY_RANK_R8
    hits = np.zeros(8, np.int64)
    q = hit_aware_quota_rank_cost(hits, 12, 8, H2D_US_BY_RANK_R8)
    assert q.sum() == 12 and q.max() <= 2
    assert np.array_equal(q, [1, 1, 1, 1, 2, 2, 2, 2])
    # A hit-heavy fast rank (40 hits = 2.84 ms) never becomes the cheapest rank.
    hits = np.array([0, 0, 0, 0, 40, 0, 0, 0], np.int64)
    q = hit_aware_quota_rank_cost(hits, 12, 8, H2D_US_BY_RANK_R8)
    assert q.sum() == 12 and q.max() <= 2 and q[4] == 0


def test_haq_fast_policy_runs_in_controller_step():
    p = Policy([16] * 8, np.zeros((48, 128, 128), np.float32), False, 21)
    selected = np.repeat(np.arange(12, dtype=np.int16)[:, None], 2, axis=1)
    weights = np.full((12, 2), .5, dtype=np.float32)
    origins = np.arange(12, dtype=np.int64) % 8
    result = p.apply(0, selected, weights, origins,
                     np.zeros(128, np.float32), np.zeros((128, 8), np.int32))
    counts = np.bincount([row[0] for row in result[5]], minlength=8)
    assert np.array_equal(counts, [1, 1, 1, 1, 2, 2, 2, 2])


def test_placement_controls_respect_quota_and_order():
    from placement_controls import worst_assignment_with_quota, random_assignment_with_quota
    from la_placement import load_locality_near_assignment
    rng = np.random.default_rng(3)
    demand = rng.integers(0, 6, size=(128, 8)).astype(np.int64)
    owner = np.zeros(48 * 128, np.int16)
    misses = np.arange(20, dtype=np.int64)
    quota = np.array([2, 2, 2, 2, 3, 3, 3, 3], np.int64)
    near = load_locality_near_assignment(demand, misses, owner, 0, 200, quota)
    worst = worst_assignment_with_quota(demand, misses, owner, 0, quota)
    np.random.seed(5)
    rand = random_assignment_with_quota(misses, quota)
    for a in (near, worst, rand):
        assert np.array_equal(np.bincount(a, minlength=8), quota)
    rows = lambda a: np.bincount(a, weights=demand[misses].sum(1), minlength=8)
    assert rows(worst).max() > rows(near).max()
    local = lambda a: sum(int(demand[e, r]) for e, r in zip(misses, a))
    assert local(worst) < local(near)


def test_fast_random_policy_is_seeded_per_event():
    quota = table([7, 6, 5, 4, 3, 2, 1, 0])
    sim = np.zeros((48, 128, 128), np.float32)
    selected = np.repeat(np.arange(12, dtype=np.int16)[:, None], 2, axis=1)
    weights = np.full((12, 2), .5, dtype=np.float32)
    origins = np.arange(12, dtype=np.int64) % 8
    outs = []
    for _ in range(2):
        p = Policy([16] * 8, sim, False, 23, quota_lut=quota)
        r = p.apply(0, selected, weights, origins, np.zeros(128, np.float32), np.zeros((128, 8), np.int32))
        outs.append([f[:2] for f in r[5]])
        assert np.array_equal(np.bincount([f[0] for f in r[5]], minlength=8), quota[12])
    assert outs[0] == outs[1]
