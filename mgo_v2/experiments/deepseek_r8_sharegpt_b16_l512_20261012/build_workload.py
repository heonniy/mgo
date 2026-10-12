"""Freeze the DeepSeek-V2-Lite ShareGPT R8/C30/local-B16/L512/O64 workload.

Mirrors qwen_r8_sharegpt_b16_l512_20261009: first 128 target and first 128 disjoint
warmup requests of the frozen DeepSeek ShareGPT 256-request pool (R4/B64 cell),
same exact input tokens, reassigned to R8 local B16 (origin_rank = index // 16).
The pool shares conversations with the Qwen pool (main_table_2x2_20261008), which
this script verifies against the Qwen R8 target.
"""
import hashlib
import json
from pathlib import Path

SRC = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json')
QWEN_R8 = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/Qwen3_ShareGPT_R8_C30_B16_L512_O64_target.json')
OUT = Path('/home/hwlee/mgo-results/deepseek_r8_sharegpt_b16_l512_20261012')
CELL = 'DeepSeekV2Lite_ShareGPT_R8_C30_B16_L512_O64'
WORLD, BATCH, PERCENT = 8, 16, 30


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    src = json.loads(SRC.read_text())
    cell = next(c for c in src['cells'] if c['cell'] == 'DeepSeekV2Lite_ShareGPT_R4_C30_B64_L512_O64')
    OUT.mkdir(parents=True, exist_ok=True)
    out_cell = {k: cell[k] for k in ('dataset', 'model', 'model_path', 'input_tokens', 'output_tokens')}
    slots = 26 * 64 * PERCENT // 100
    per_rank = [slots // WORLD + (r < slots % WORLD) for r in range(WORLD)]
    out_cell.update(cell=CELL, local_batch=BATCH, global_requests=WORLD * BATCH, cache_percent=PERCENT,
                    expert_slots=slots, expert_slots_per_rank=per_rank,
                    expert_budget_bytes=slots * 8650752 * 2)
    ids = {}
    for phase in ('target', 'warmup'):
        assert sha(cell[phase]['path']) == cell[phase]['sha256']
        data = json.loads(Path(cell[phase]['path']).read_text())
        rows = data['requests'][:WORLD * BATCH]
        assert len(rows) == WORLD * BATCH and all(len(r['input_ids']) == 512 for r in rows)
        new = []
        for i, r in enumerate(rows):
            r = dict(r); r['global_index'] = i; r['origin_rank'] = i // BATCH; new.append(r)
        ids[phase] = {r['conversation_id'] for r in new}
        path = OUT / f'{CELL}_{phase}.json'
        path.write_text(json.dumps(dict(cell=CELL, phase=phase, global_requests=len(new),
                                        input_tokens=512, output_tokens=64, requests=new)) + '\n')
        out_cell[phase] = dict(path=str(path), sha256=sha(path))
    assert not ids['target'] & ids['warmup']
    qwen = {r['conversation_id'] for r in json.loads(QWEN_R8.read_text())['requests']}
    manifest = dict(status='FROZEN', physical_gpus=list(range(WORLD)), dataset='ShareGPT', output_tokens=64,
                    selection='First 128 target and first 128 disjoint warmup requests of the frozen DeepSeek '
                              'ShareGPT 256-request pool (DeepSeekV2Lite_ShareGPT_R4_C30_B64_L512_O64); same exact '
                              'input tokens, reassigned to R8 local B16.',
                    same_conversations_as_qwen_r8_target=ids['target'] == qwen,
                    cells=[out_cell])
    (OUT / 'WORKLOADS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(cell=CELL, slots=slots, per_rank=per_rank,
                          same_conversations_as_qwen_r8=ids['target'] == qwen,
                          manifest_sha256=sha(OUT / 'WORKLOADS.json')), indent=1))


if __name__ == '__main__':
    main()
