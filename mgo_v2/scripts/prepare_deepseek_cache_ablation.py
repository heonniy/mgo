"""Reuse the frozen main-table ShareGPT requests across DeepSeek cache sizes."""

import argparse
import hashlib
import json
from pathlib import Path


SOURCE = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json')
ROOT = Path('/home/hwlee/mgo-results/deepseek_cache_ablation_20261009')
RECEIPT = (Path(__file__).resolve().parents[1] /
           'experiments/deepseek_cache_ablation_20261009/FROZEN_MANIFEST_RECEIPT.json')
EXPERT_BYTES = 3 * 2048 * 1408 * 2
TOTAL_EXPERTS = 26 * 64


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(batches):
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'FROZEN' and source['dataset'] == 'ShareGPT'
    selected = {}
    for batch in batches:
        matches = [cell for cell in source['cells'] if cell['model'] == 'DeepSeekV2Lite'
                   and cell['input_tokens'] == 512 and cell['local_batch'] == batch]
        assert len(matches) == 1
        selected[batch] = matches[0]

    cells = []
    for batch in batches:
        old = selected[batch]
        requests = {}
        for phase in ('warmup', 'target'):
            path = Path(old[phase]['path'])
            assert sha256(path) == old[phase]['sha256']
            data = json.loads(path.read_text())
            assert len(data['requests']) == 4 * batch
            assert all(len(row['input_ids']) == 512 for row in data['requests'])
            requests[phase] = data['requests']
        assert set(row['request_id'] for row in requests['warmup']).isdisjoint(
            row['request_id'] for row in requests['target'])
        for percent in (20, 30, 40, 50):
            slots = TOTAL_EXPERTS * percent // 100
            per_rank = [slots // 4 + (rank < slots % 4) for rank in range(4)]
            cell_name = f'DeepSeekV2Lite_ShareGPT_R4_C{percent}_B{batch}_L512_O64'
            phase_receipts = {}
            for phase in ('warmup', 'target'):
                path = ROOT / 'manifests' / f'{cell_name}_{phase}.json'
                write(path, dict(cell=cell_name, phase=phase, global_requests=4 * batch,
                                 input_tokens=512, output_tokens=64,
                                 requests=requests[phase]))
                phase_receipts[phase] = dict(path=str(path), sha256=sha256(path))
            cells.append(dict(cell=cell_name, dataset='ShareGPT',
                              model='DeepSeekV2Lite', model_path=old['model_path'],
                              local_batch=batch, global_requests=4 * batch,
                              input_tokens=512, output_tokens=64,
                              cache_percent=percent, expert_slots=slots,
                              expert_slots_per_rank=per_rank,
                              expert_budget_bytes=slots * EXPERT_BYTES,
                              **phase_receipts))

    manifest_path = ROOT / 'WORKLOADS.json'
    write(manifest_path, dict(status='FROZEN', dataset='ShareGPT',
                              model_paths={'DeepSeekV2Lite': selected[batches[0]]['model_path']},
                              source_manifest=str(SOURCE), source_manifest_sha256=sha256(SOURCE),
                              selection='Identical frozen main-table ShareGPT DeepSeek input-512 '
                                        'requests and rank order at every cache percentage; '
                                        'disjoint warmup and target batches.',
                              physical_gpus=[0, 1, 4, 5], output_tokens=64, cells=cells))
    write(RECEIPT, dict(status='PASS', manifest_path=str(manifest_path),
                        manifest_sha256=sha256(manifest_path), source_manifest_sha256=sha256(SOURCE),
                        cells=[dict(cell=cell['cell'], cache_percent=cell['cache_percent'],
                                    local_batch=cell['local_batch'],
                                    expert_slots_per_rank=cell['expert_slots_per_rank'],
                                    warmup_sha256=cell['warmup']['sha256'],
                                    target_sha256=cell['target']['sha256']) for cell in cells]))
    print(f'PASS {len(cells)} frozen workload cells: {manifest_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, choices=(16, 64), action='append')
    args = parser.parse_args()
    main(tuple(args.batch or (16, 64)))
