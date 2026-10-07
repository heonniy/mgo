"""Freeze private two-model headline prompts; commit only the hash receipt."""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path('/home/hwlee/mgo-results/main_table_2x2_20261008')
REPO = Path(__file__).resolve().parents[1] / 'experiments/main_table_2x2_20261008'
SHAREGPT = Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/sources/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json')
LMSYS = Path('/home/hwlee/dataset')
MODELS = {
    'Qwen3': Path('/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'),
    'DeepSeekV2Lite': Path('/home/hwlee/model/DeepSeek-V2-Lite-Chat'),
}
EXPERT_BUDGETS = {
    'Qwen3': dict(expert_slots=1843, expert_slots_per_rank=[461, 461, 461, 460],
                  expert_bytes=9 * 2**20),
    # 26 routed MoE layers after the first dense layer, 64 experts/layer.
    # Each BF16 expert has gate/up/down projections of 2048 x 1408.
    'DeepSeekV2Lite': dict(expert_slots=499, expert_slots_per_rank=[125, 125, 125, 124],
                           expert_bytes=3 * 2048 * 1408 * 2),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def final_user_prefix(conversation, style):
    prefix = []
    best = None
    for item in conversation or []:
        if not isinstance(item, dict):
            break
        if style == 'sharegpt':
            role = {'human': 'user', 'gpt': 'assistant'}.get(item.get('from'))
            content = item.get('value')
        else:
            role = item.get('role')
            content = item.get('content')
        if role not in ('user', 'assistant') or not isinstance(content, str):
            break
        prefix.append({'role': role, 'content': content})
        if role == 'user' and content.strip():
            best = prefix.copy()
    return best


def source_rows(dataset):
    if dataset == 'ShareGPT':
        for index, row in enumerate(json.loads(SHAREGPT.read_text())):
            yield index, row.get('id'), final_user_prefix(row.get('conversations'), 'sharegpt')
    else:
        paths = sorted(LMSYS.glob('train-*-of-00006-*.parquet'))
        if len(paths) != 6:
            raise RuntimeError(f'expected six official LMSYS shards, found {len(paths)}')
        index = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=256, columns=['conversation_id', 'conversation', 'redacted']):
                for row in batch.to_pylist():
                    if row.get('redacted') is not True:
                        yield index, row.get('conversation_id'), final_user_prefix(row.get('conversation'), 'lmsys')
                    index += 1


def main(dataset):
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
                  for name, path in MODELS.items()}
    selected = []
    seen = set()
    scanned = 0
    for source_row, conversation_id, messages in source_rows(dataset):
        scanned += 1
        if not conversation_id or conversation_id in seen or not messages:
            continue
        ids = {}
        try:
            for name, tokenizer in tokenizers.items():
                ids[name] = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    **({'enable_thinking': False} if name == 'Qwen3' else {}))
        except (TypeError, ValueError):
            continue
        if min(map(len, ids.values())) < 1024:
            continue
        seen.add(conversation_id)
        selected.append((source_row, conversation_id, ids))
        if len(selected) == 512:
            break
    if len(selected) != 512:
        raise RuntimeError(f'{dataset}: only {len(selected)} eligible distinct conversations after {scanned} rows')
    cells = []
    for model_name, model_path in MODELS.items():
        budget = EXPERT_BUDGETS[model_name]
        assert sum(budget['expert_slots_per_rank']) == budget['expert_slots']
        for length in (512, 1024):
            for batch in (16, 64):
                count = 4 * batch
                cell = f'{model_name}_{dataset}_R4_C30_B{batch}_L{length}_O64'
                phases = {}
                for phase, chosen in (('target', selected[:count]), ('warmup', selected[256:256 + count])):
                    requests = []
                    for index, (source_row, conversation_id, ids) in enumerate(chosen):
                        token_ids = ids[model_name][-length:]
                        assert len(token_ids) == length
                        requests.append(dict(request_id=source_row, source_row=source_row,
                                             conversation_id=conversation_id, global_index=index,
                                             origin_rank=index // batch, input_ids=token_ids,
                                             input_tokens=length))
                    path = ROOT / dataset / model_name / 'manifests' / f'{cell}_{phase}.json'
                    write(path, dict(cell=cell, phase=phase, global_requests=count,
                                     input_tokens=length, output_tokens=64, requests=requests))
                    phases[phase] = dict(path=str(path), sha256=digest(path))
                cells.append(dict(cell=cell, dataset=dataset, model=model_name,
                                  model_path=str(model_path), local_batch=batch,
                                  global_requests=count, input_tokens=length,
                                  output_tokens=64, cache_percent=30,
                                  expert_slots=budget['expert_slots'],
                                  expert_slots_per_rank=budget['expert_slots_per_rank'],
                                  expert_budget_bytes=budget['expert_slots'] * budget['expert_bytes'],
                                  **phases))
    manifest = ROOT / dataset / 'WORKLOADS.json'
    write(manifest, dict(status='FROZEN', dataset=dataset, model_paths={k: str(v) for k, v in MODELS.items()},
                         selection='First 512 distinct, non-redacted conversations with a final user turn and >=1024 rendered tokens under both model chat templates; target 0:256, disjoint warmup 256:512; use final exact 512/1024 tokens.',
                         physical_gpus=[0, 1, 4, 5], output_tokens=64,
                         scanned_source_rows=scanned, eligible_conversations=512, cells=cells))
    source_paths = [SHAREGPT] if dataset == 'ShareGPT' else sorted(LMSYS.glob('train-*-of-00006-*.parquet'))
    write(REPO / f'{dataset.upper().replace("-", "_")}_MANIFEST_RECEIPT.json',
          dict(status='PASS', dataset=dataset, source_files=[dict(name=p.name, sha256=digest(p)) for p in source_paths],
               scanned_source_rows=scanned, eligible_conversations=512, model_names=list(MODELS),
               cells=[dict(cell=c['cell'], target_sha256=c['target']['sha256'],
                           warmup_sha256=c['warmup']['sha256']) for c in cells],
               private_manifest_path=str(manifest), private_manifest_sha256=digest(manifest)))
    print(f'PASS {dataset}: {len(cells)} workload cells, {scanned} source rows scanned', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('ShareGPT', 'LMSYS-Chat-1M'), required=True)
    main(parser.parse_args().dataset)
