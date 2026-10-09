"""Freeze a two-rank B16 subset of the existing R8 Qwen ShareGPT workload."""

import hashlib
import json
from pathlib import Path


SOURCE = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/WORKLOADS.json')
ROOT = Path('/home/hwlee/mgo-results/qwen_r2_sharegpt_b16_l512_20261010')
CELL = 'Qwen3_ShareGPT_R2_C30_B16_L512_O64'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'FROZEN' and source['physical_gpus'] == list(range(8))
    prior = source['cells'][0]
    assert prior['local_batch'] == 16 and prior['input_tokens'] == 512
    assert prior['output_tokens'] == 64 and prior['expert_slots'] == 1843
    cell = dict(prior, cell=CELL, global_requests=32,
                expert_slots_per_rank=[922, 921])
    for phase in ('warmup', 'target'):
        receipt = prior[phase]
        path = Path(receipt['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
        selected = json.loads(path.read_text())['requests'][:32]
        assert len(selected) == 32
        assert [row['origin_rank'] for row in selected] == [0] * 16 + [1] * 16
        cell[phase] = write(ROOT / f'{CELL}_{phase}.json',
                            {'cell': CELL, 'phase': phase, 'global_requests': 32,
                             'input_tokens': 512, 'output_tokens': 64,
                             'requests': selected})
    warm_ids = {row['request_id'] for row in json.loads(Path(cell['warmup']['path']).read_text())['requests']}
    target_ids = {row['request_id'] for row in json.loads(Path(cell['target']['path']).read_text())['requests']}
    assert not warm_ids & target_ids
    write(ROOT / 'WORKLOADS.json', {'status': 'FROZEN', 'physical_gpus': [0, 1],
                                    'dataset': 'ShareGPT', 'output_tokens': 64,
                                    'selection': 'First 32 rank-ordered requests of each disjoint frozen R8 phase',
                                    'source_manifest_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                                    'cells': [cell]})


if __name__ == '__main__':
    main()
