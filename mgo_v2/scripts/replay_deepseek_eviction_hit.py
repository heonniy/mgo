"""CPU-only LFU/LRU/gate replay from the existing DeepSeek D1 route capture."""

import hashlib
import json
from pathlib import Path

import numpy as np

from main_eviction_history_replay import POLICIES, replay


ROOT = Path(__file__).resolve().parents[1]
ROUTES = Path('/home/hwlee/mgo-results/deepseek_cache_ablation_20261009/critical_path_d1/route_capture_v1')
REFERENCE = Path('/home/hwlee/mgo-results/headline_r4_20261007/dca_d1_fixedroute_c20_pair1_v1')
OUT = ROOT / 'experiments/deepseek_eviction_hit_20261009'
LAYERS, EXPERTS, TOPK, BATCH, WORLD, STEPS = 26, 64, 6, 16, 4, 63


def read(path):
    return json.loads(path.read_text())


def validated_files():
    filenames = [ROUTES / f'route_rank{r}.npz' for r in range(WORLD)]
    hashes = []
    for rank, path in enumerate(filenames):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = read(REFERENCE / f'repeat1_rank{rank}.json')
        assert receipt['route_mode'] == 'replay' and receipt['route_sha256'] == digest
        hashes.append(digest)
    assert read(REFERENCE / 'status.json')['status'] == 'PASS'
    return filenames, hashes


def pack(filenames):
    counts = np.empty(((STEPS + 1) * LAYERS, EXPERTS), dtype=np.int32)
    gates = np.empty_like(counts, dtype=np.float32)
    selected = []
    probs = []
    for path in filenames:
        with np.load(path, allow_pickle=False) as data:
            assert data['prefill_selected'].shape == (LAYERS, BATCH * 512, TOPK)
            assert data['prefill_probs'].shape == (LAYERS, BATCH * 512, EXPERTS)
            assert data['decode_selected'].shape == (STEPS, LAYERS, BATCH, TOPK)
            assert data['decode_probs'].shape == (STEPS, LAYERS, BATCH, EXPERTS)
            selected.append((data['prefill_selected'].copy(), data['decode_selected'].copy()))
            probs.append((data['prefill_probs'].copy(), data['decode_probs'].copy()))
    history = [np.empty((0, EXPERTS), dtype=np.float64) for _ in range(LAYERS)]
    for step in range(STEPS + 1):
        for layer in range(LAYERS):
            event = step * LAYERS + layer
            ids = [selected[r][0][layer] if step == 0 else selected[r][1][step - 1, layer]
                   for r in range(WORLD)]
            flat = np.concatenate(ids).ravel()
            assert flat.min() >= 0 and flat.max() < EXPERTS
            counts[event] = np.bincount(flat, minlength=EXPERTS)
            assert counts[event].sum() == (BATCH * 512 if step == 0 else BATCH) * WORLD * TOPK
            # Global rank order; prefill only its final 128 router rows survive.
            row = (probs[-1][0][layer, -128:] if step == 0 else
                   np.concatenate([probs[r][1][step - 1, layer] for r in range(WORLD)]))
            assert np.isfinite(row).all()
            history[layer] = np.concatenate((history[layer], row.astype(np.float64)))[-128:]
            gates[event] = history[layer].mean(axis=0).astype(np.float32)
    return counts, gates


def capacities(percent):
    total = LAYERS * EXPERTS * percent // 100
    physical = [total // WORLD + (r < total % WORLD) for r in range(WORLD)]
    # The DeepSeek runtime reserves two non-MAIN slots on each rank.
    return np.array([n - 2 for n in physical], dtype=np.int32)


def main():
    filenames, hashes = validated_files()
    counts, gates = pack(filenames)
    OUT.mkdir(exist_ok=True)
    rows = []
    for percent in (30, 60):
        cap = capacities(percent)
        policies = []
        reference_lru = None
        for mode, name in enumerate(POLICIES):
            metrics, slots, _, _ = replay(counts, gates, cap, mode, 42, LAYERS)
            assert np.array_equal(metrics[:, :2].sum(axis=1), (counts > 0).sum(axis=1))
            assert np.all(metrics[:, 3] <= metrics[:, 1])
            assert int(metrics[:, 1].sum() - metrics[:, 2].sum()) == int((slots >= 0).sum())
            if name == 'LRU-reset':
                reference_lru = metrics
            if name == 'LRU-cumulative':
                assert np.array_equal(metrics, reference_lru)
            decode = metrics[LAYERS:].sum(axis=0)
            fields = ['hit', 'miss', 'eviction', 'reload']
            values = dict(zip(fields, map(int, decode)))
            values['hit_rate'] = values['hit'] / (values['hit'] + values['miss'])
            policies.append(dict(policy=name, decode=values))
            print(f'C{percent} {name}: {values["hit_rate"]:.4%}', flush=True)
        rows.append(dict(cache_percent=percent, capacities=cap.tolist(), policies=policies))
    result = dict(status='PASS', model='DeepSeek-V2-Lite-Chat',
                  workload='ShareGPT B16/rank input512 63 decode forwards',
                  source='previously captured D1 frozen routing trace',
                  source_files=list(map(str, filenames)), source_sha256=hashes,
                  cpu_counterfactual=True, admission='BR, seed42, common to all policies',
                  gate='rolling last 128 full-router probability rows per layer',
                  prefetch=False, main_only=True,
                  caveat='No source-policy parity or physical timing claim; C60 is counterfactual.',
                  results=rows)
    (OUT / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
