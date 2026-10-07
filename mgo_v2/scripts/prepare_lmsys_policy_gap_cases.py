"""Nominate LMSYS-Chat-1M C60 physical cases from its input128 proxy."""
import json
from pathlib import Path

import numpy as np
from ca_stress_cpu import proxy_scores
from prepare_policy_gap_c60 import ROOT, sha, write

SOURCE = ROOT / 'LMSYS_Chat_1M_requests.json'
PROXY = ROOT / 'LMSYS_proxy_routes_input128.npy'


def main():
    source = json.loads(SOURCE.read_text())
    rows = source['requests']
    routes = np.load(PROXY, mmap_mode='r')
    assert source['status'] == 'PASS' and len(rows) == 1024 and routes.shape == (48, 1024, 8)
    cells = []
    for batch in (8, 16, 64):
        n = 4 * batch
        scored = []
        for sample_seed in range(24):
            perm = np.random.default_rng(sample_seed).permutation(len(rows))
            sample = perm[:n]
            orders = np.stack([np.random.default_rng(dp_seed).permutation(n) for dp_seed in range(8)])
            values = proxy_scores(routes, sample, orders, 4, batch)
            scored.extend(dict(sample_seed=sample_seed, rank_order_seed=dp_seed,
                               proxy_locality_gain=float(value)) for dp_seed, value in enumerate(values))
        scored.sort(key=lambda x: (-x['proxy_locality_gain'], x['sample_seed'], x['rank_order_seed']))
        selected = []
        for candidate in scored:
            if len(selected) == 2:
                break
            if any(x['sample_seed'] == candidate['sample_seed'] for x in selected):
                continue
            selected.append(candidate)
        assert len(selected) == 2
        for case_id, seed in enumerate(selected):
            permutation = np.random.default_rng(seed['sample_seed']).permutation(len(rows))
            order = np.random.default_rng(seed['rank_order_seed']).permutation(n)
            target = permutation[:n][order]
            warmup = permutation[n:2*n]
            assert not set(target) & set(warmup)
            cell = f'LMSYS_R4_C60_B{batch}_L128_O33_s{seed["sample_seed"]}_d{seed["rank_order_seed"]}'
            phases = {}
            for phase, indices in [('target', target), ('warmup', warmup)]:
                requests = []
                for position, index in enumerate(indices):
                    row = rows[int(index)]
                    requests.append(dict(**row, global_index=position, origin_rank=position // batch))
                path = ROOT / 'manifests' / f'{cell}_{phase}.json'
                write(path, dict(status='FROZEN', dataset='LMSYS-Chat-1M', cell=cell, phase=phase,
                                 input_tokens=128, decode_forwards=32, requests=requests))
                phases[phase] = dict(path=str(path), sha256=sha(path))
            cells.append(dict(cell=cell, dataset='LMSYS-Chat-1M', local_batch=batch, global_requests=n,
                              input_tokens=128, output_tokens=33, decode_forwards=32,
                              expert_slots_per_rank=[922, 922, 921, 921], seed=seed, case_id=case_id, **phases))
        print(f'B{batch}: ' + ', '.join(f"s{x['sample_seed']}/d{x['rank_order_seed']} proxy={x['proxy_locality_gain']:.4f}" for x in selected), flush=True)
    write(ROOT / 'LMSYS_WORKLOADS.json', dict(status='PASS', dataset='LMSYS-Chat-1M', source=str(SOURCE),
                                             source_sha256=sha(SOURCE), proxy=str(PROXY), proxy_sha256=sha(PROXY),
                                             proxy_limit='input128 prefill last-token routing, not full32-decode runtime or TPOT',
                                             sample_seeds=list(range(24)), rank_order_seeds=list(range(8)), cells=cells))


if __name__ == '__main__':
    main()
