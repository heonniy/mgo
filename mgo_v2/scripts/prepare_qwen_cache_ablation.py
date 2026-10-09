"""Freeze the existing Qwen ShareGPT B16/L512 requests at four cache sizes."""

import hashlib
import json
from pathlib import Path


SOURCE = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json')
ROOT = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009')
REPO = (Path(__file__).resolve().parents[1] /
        'experiments/qwen_cache_ablation_20261009')
EXPERT_BYTES = 9 * 2**20
TOTAL_EXPERTS = 48 * 128
PERCENTS = (20, 30, 40, 50)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'FROZEN' and source['dataset'] == 'ShareGPT'
    matches = [cell for cell in source['cells'] if cell['model'] == 'Qwen3'
               and cell['local_batch'] == 16 and cell['input_tokens'] == 512
               and cell['output_tokens'] == 64]
    assert len(matches) == 1
    original = matches[0]
    requests = {}
    for phase in ('warmup', 'target'):
        path = Path(original[phase]['path'])
        assert digest(path) == original[phase]['sha256']
        content = json.loads(path.read_text())
        assert len(content['requests']) == 64
        assert all(len(row['input_ids']) == 512 for row in content['requests'])
        requests[phase] = content['requests']
    assert {r['request_id'] for r in requests['warmup']}.isdisjoint(
        r['request_id'] for r in requests['target'])
    cells = []
    for percent in PERCENTS:
        slots = TOTAL_EXPERTS * percent // 100
        per_rank = [slots // 4 + (rank < slots % 4) for rank in range(4)]
        name = f'Qwen3_ShareGPT_R4_C{percent}_B16_L512_O64'
        phases = {}
        for phase in ('warmup', 'target'):
            path = ROOT / 'manifests' / f'{name}_{phase}.json'
            write(path, dict(cell=name, phase=phase, global_requests=64,
                             input_tokens=512, output_tokens=64,
                             requests=requests[phase]))
            phases[phase] = dict(path=str(path), sha256=digest(path))
        cells.append(dict(cell=name, dataset='ShareGPT', model='Qwen3',
                          model_path=original['model_path'], local_batch=16,
                          global_requests=64, input_tokens=512, output_tokens=64,
                          cache_percent=percent, expert_slots=slots,
                          expert_slots_per_rank=per_rank,
                          expert_budget_bytes=slots * EXPERT_BYTES,
                          **phases))
    manifest = ROOT / 'WORKLOADS.json'
    write(manifest, dict(status='FROZEN', dataset='ShareGPT',
                         model_paths={'Qwen3': original['model_path']},
                         source_manifest=str(SOURCE),
                         source_manifest_sha256=digest(SOURCE),
                         selection='Identical Qwen main-table ShareGPT B16/L512 '
                                   'requests and rank order at every cache percentage; '
                                   'warmup and target batches are disjoint.',
                         physical_gpus=[0, 1, 4, 5], output_tokens=64,
                         cells=cells))
    write(REPO / 'FROZEN_MANIFEST_RECEIPT.json',
          dict(status='PASS', manifest_path=str(manifest),
               manifest_sha256=digest(manifest),
               source_manifest_sha256=digest(SOURCE),
               cells=[dict(cell=c['cell'], cache_percent=c['cache_percent'],
                           expert_slots=c['expert_slots'],
                           expert_slots_per_rank=c['expert_slots_per_rank'],
                           warmup_sha256=c['warmup']['sha256'],
                           target_sha256=c['target']['sha256']) for c in cells]))
    print(f'PASS {len(cells)} frozen Qwen workload cells: {manifest}')


if __name__ == '__main__':
    main()
