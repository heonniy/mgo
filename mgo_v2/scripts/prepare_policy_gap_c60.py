"""Bounded ShareGPT seed screen for the R4/C60/input128/decode32 policy study.

The historical variable-context routes are used only to nominate workloads.
Physical results are measured on freshly executed 128-token prompts with a
common frozen route; the screen is never described as an exact performance
oracle or as a global maximum.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from ca_stress_cpu import proxy_scores

SOURCE = Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/ShareGPT_requests.json')
ROUTES = Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/pool/ShareGPT/proxy_routes.npy')
ROOT = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    source = json.loads(SOURCE.read_text())
    all_rows = source['requests']
    seen = set()
    eligible = []
    for row in all_rows:
        if len(row['input_ids']) < 128 or row['conversation_id'] in seen:
            continue
        seen.add(row['conversation_id'])
        eligible.append(row)
    assert len(eligible) >= 512
    pool_indices = np.asarray([r['request_id'] for r in eligible], dtype=np.int32)
    routes = np.asarray(np.load(ROUTES, mmap_mode='r')[:, pool_indices], dtype=np.uint8)
    assert routes.shape == (48, len(eligible), 8)
    meta = dict(status='PASS', dataset='ShareGPT', source=str(SOURCE), source_sha256=sha(SOURCE),
                proxy_source=str(ROUTES), proxy_source_sha256=sha(ROUTES), eligible=len(eligible),
                prompt_rule='last 128 existing rendered tokens; no padding',
                proxy_limit='historical variable-context routes nominate seeds only; physical input128 reruns decide gains',
                sample_seeds=list(range(24)), rank_order_seeds=list(range(8)))
    cells = []
    for batch in (8, 16, 64):
        n = 4 * batch
        scored = []
        for sample_seed in range(24):
            perm = np.random.default_rng(sample_seed).permutation(len(eligible))
            sample = perm[:n]
            orders = np.stack([np.random.default_rng(dp_seed).permutation(n) for dp_seed in range(8)])
            proxy = proxy_scores(routes, sample, orders, 4, batch)
            for dp_seed, value in enumerate(proxy):
                scored.append(dict(sample_seed=sample_seed, rank_order_seed=dp_seed, proxy_locality_gain=float(value)))
        scored.sort(key=lambda x: (-x['proxy_locality_gain'], x['sample_seed'], x['rank_order_seed']))
        selected = []
        for row in scored:
            if len(selected) == 2:
                break
            if any(row['sample_seed'] == old['sample_seed'] for old in selected):
                continue
            selected.append(row)
        assert len(selected) == 2
        cases = []
        for case_id, seed in enumerate(selected):
            perm = np.random.default_rng(seed['sample_seed']).permutation(len(eligible))
            order = np.random.default_rng(seed['rank_order_seed']).permutation(n)
            target_indices = perm[:n][order]
            warmup_indices = perm[n:2*n]
            assert not set(target_indices) & set(warmup_indices)
            case = f'ShareGPT_R4_C60_B{batch}_L128_O33_s{seed["sample_seed"]}_d{seed["rank_order_seed"]}'
            phases = {}
            for phase, indices in [('target', target_indices), ('warmup', warmup_indices)]:
                requests = []
                for pos, ix in enumerate(indices):
                    original = eligible[int(ix)]
                    requests.append(dict(request_id=int(original['request_id']), source_row=original['source_row'],
                                         conversation_id=original['conversation_id'], global_index=pos,
                                         origin_rank=pos // batch, input_ids=original['input_ids'][-128:], input_tokens=128))
                path = ROOT / 'manifests' / f'{case}_{phase}.json'
                write(path, dict(status='FROZEN', dataset='ShareGPT', case=case, phase=phase,
                                 input_tokens=128, decode_forwards=32, requests=requests))
                phases[phase] = dict(path=str(path), sha256=sha(path))
            cases.append(dict(cell=case, dataset='ShareGPT', local_batch=batch, global_requests=n,
                              input_tokens=128, output_tokens=33, decode_forwards=32,
                              expert_slots_per_rank=[922, 922, 921, 921], seed=seed, case_id=case_id, **phases))
        cells.extend(cases)
        print(f'B{batch}: ' + ', '.join(f"s{x['sample_seed']}/d{x['rank_order_seed']} proxy={x['proxy_locality_gain']:.4f}" for x in selected), flush=True)
    meta['cells'] = cells
    write(ROOT / 'WORKLOADS.json', meta)
    print(f'PASS {ROOT / "WORKLOADS.json"}', flush=True)


if __name__ == '__main__':
    main()
