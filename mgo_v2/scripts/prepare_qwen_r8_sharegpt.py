"""Freeze the Qwen ShareGPT R8/B16/L512 requests from the prior source pool."""

import hashlib
import json
from pathlib import Path


OLD = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json')
ROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
RECEIPT = Path(__file__).resolve().parents[1] / 'experiments/qwen_r8_sharegpt_b16_l512_20261009/WORKLOAD_RECEIPT.json'
CELL = 'Qwen3_ShareGPT_R8_C30_B16_L512_O64'
WORLD = 8
BATCH = 16


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads(OLD.read_text())
    source = next(c for c in old['cells'] if c['cell'] == 'Qwen3_ShareGPT_R4_C30_B64_L512_O64')
    assert old['status'] == 'FROZEN' and source['global_requests'] == 256
    phases = {}
    request_ids = {}
    for phase in ('target', 'warmup'):
        old_path = Path(source[phase]['path'])
        assert digest(old_path) == source[phase]['sha256']
        previous = json.loads(old_path.read_text())['requests']
        selected = []
        for index, row in enumerate(previous[:WORLD * BATCH]):
            assert len(row['input_ids']) == 512
            selected.append(dict(row, global_index=index, origin_rank=index // BATCH))
        assert len(selected) == WORLD * BATCH
        request_ids[phase] = [row['request_id'] for row in selected]
        path = ROOT / f'{CELL}_{phase}.json'
        write(path, dict(cell=CELL, phase=phase, global_requests=WORLD * BATCH,
                         input_tokens=512, output_tokens=64, requests=selected))
        phases[phase] = dict(path=str(path), sha256=digest(path))
    assert len(set(request_ids['target'] + request_ids['warmup'])) == 2 * WORLD * BATCH
    slots = [1843 // WORLD + (rank < 1843 % WORLD) for rank in range(WORLD)]
    assert sum(slots) == 1843
    cell = dict(cell=CELL, dataset='ShareGPT', model='Qwen3',
                model_path='/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507',
                local_batch=BATCH, global_requests=WORLD * BATCH,
                input_tokens=512, output_tokens=64, cache_percent=30,
                expert_slots=1843, expert_slots_per_rank=slots,
                expert_budget_bytes=1843 * 9 * 2**20, **phases)
    manifest = ROOT / 'WORKLOADS.json'
    write(manifest, dict(status='FROZEN', physical_gpus=list(range(WORLD)),
                         dataset='ShareGPT', output_tokens=64,
                         selection='First 128 target and first 128 disjoint warmup requests '
                                   'from the prior frozen Qwen ShareGPT 256-request pool; '
                                   'same exact input tokens, reassigned to R8 local B16.',
                         cells=[cell]))
    write(RECEIPT, dict(status='PASS', cell=CELL, world=WORLD, local_batch=BATCH,
                        target_sha256=phases['target']['sha256'],
                        warmup_sha256=phases['warmup']['sha256'],
                        manifest_sha256=digest(manifest),
                        old_manifest_sha256=digest(OLD),
                        source_cell=source['cell'],
                        source_target_sha256=source['target']['sha256'],
                        source_warmup_sha256=source['warmup']['sha256'],
                        disjoint_request_ids=True, expert_slots_per_rank=slots))
    print(manifest)


if __name__ == '__main__':
    main()
