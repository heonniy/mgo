"""Freeze a bounded ShareGPT B16 sample-seed list before GPU screening."""

import copy
import hashlib
import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/Qwen3/manifests')
BASE = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
RAW = Path('/home/hwlee/mgo-results/pcie_quota_grouped_r8_b16_20261010/seeds')
CELL = 'Qwen3_ShareGPT_R8_C30_B16_L512_O64'
SEEDS = (11, 23, 37, 53)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads((SOURCE / 'Qwen3_ShareGPT_R4_C30_B64_L512_O64_target.json').read_text())
    old = json.loads((BASE / 'WORKLOADS.json').read_text())
    cell = old['cells'][0]
    assert len(source['requests']) == 256
    assert cell['global_requests'] == 128 and cell['local_batch'] == 16
    assert cell['input_tokens'] == 512 and cell['output_tokens'] == 64
    rows = []
    for seed in SEEDS:
        directory = RAW / f'seed{seed}'
        manifest_path = directory / 'WORKLOADS.json'
        if manifest_path.exists():
            raise FileExistsError(f'preserve previously frozen seed: {manifest_path}')
        rng = random.Random(seed)
        indices = rng.sample(range(256), 128)
        target = copy.deepcopy(source)
        target.update(cell=CELL, phase='target', global_requests=128)
        target['requests'] = []
        for i, source_index in enumerate(indices):
            request = copy.deepcopy(source['requests'][source_index])
            request.update(global_index=i, origin_rank=i // 16)
            target['requests'].append(request)
        target_path = directory / f'{CELL}_target.json'
        target_sha = save(target_path, target)
        new_cell = copy.deepcopy(cell)
        new_cell['target'] = {'path': str(target_path), 'sha256': target_sha}
        manifest = copy.deepcopy(old)
        manifest['selection'] = (f'Deterministic random.sample without replacement '
                                 f'from the frozen 256-request ShareGPT target pool; seed={seed}. '
                                 'The disjoint warmup remains fixed.')
        manifest['cells'] = [new_cell]
        manifest_sha = save(manifest_path, manifest)
        rows.append({'seed': seed, 'manifest': str(manifest_path),
                     'manifest_sha256': manifest_sha, 'target_sha256': target_sha,
                     'source_indices_sha256': hashlib.sha256(
                         json.dumps(indices).encode()).hexdigest(),
                     'warmup_sha256': cell['warmup']['sha256']})
    save(HERE / 'SEEDS.json', {'status': 'FROZEN', 'seeds': list(SEEDS),
                              'source_target': str(SOURCE / 'Qwen3_ShareGPT_R4_C30_B64_L512_O64_target.json'),
                              'candidates': rows})


if __name__ == '__main__':
    main()
