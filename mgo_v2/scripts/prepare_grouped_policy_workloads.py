"""Freeze a rank-local B8 subset alongside existing B16/B64 ShareGPT cells."""

import hashlib
import json
from pathlib import Path


SOURCE = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json')
ROOT = Path('/home/hwlee/mgo-results/grouped_policy_scaling_20261009')
B8 = 'Qwen3_ShareGPT_R4_C30_B8_L512_O64'
B16 = 'Qwen3_ShareGPT_R4_C30_B16_L512_O64'
B64 = 'Qwen3_ShareGPT_R4_C30_B64_L512_O64'


def write(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False).encode() + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()}


def main():
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'FROZEN' and source['physical_gpus'] == [0, 1, 4, 5]
    cells = {row['cell']: row for row in source['cells']}
    base = cells[B16]
    b8 = {key: value for key, value in base.items() if key not in ('cell', 'local_batch', 'global_requests', 'target', 'warmup')}
    b8.update(cell=B8, local_batch=8, global_requests=32)
    for phase in ('warmup', 'target'):
        reference = base[phase]
        payload = Path(reference['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == reference['sha256']
        data = json.loads(payload)
        requests = [data['requests'][rank*16 + index].copy()
                    for rank in range(4) for index in range(8)]
        assert all(row['origin_rank'] == index//8 for index, row in enumerate(requests))
        for index, row in enumerate(requests):
            row['global_index'] = index
        data.update(cell=B8, global_requests=32, requests=requests)
        b8[phase] = write(ROOT/'manifests'/f'{B8}_{phase}.json', data)
    manifest = dict(source)
    manifest['cells'] = [b8, cells[B16], cells[B64]]
    write(ROOT/'WORKLOADS.json', manifest)
    print(ROOT/'WORKLOADS.json')


if __name__ == '__main__':
    main()
