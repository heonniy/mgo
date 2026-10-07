"""Prepare an authorized, reproducible LMSYS-Chat-1M input128 pool.

Downloads only the first official Parquet shard after the user's gated access
is available. Raw text is not copied into the output manifest.
"""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token, hf_hub_download
from transformers import AutoTokenizer

ROOT = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')
MODEL = Path('/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507')
REPO = 'lmsys/lmsys-chat-1m'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def main():
    if not get_token():
        raise RuntimeError('LMSYS-Chat-1M gated access is not configured on this server')
    info = HfApi().dataset_info(REPO)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith('data/train-') and s.rfilename.endswith('.parquet'))
    assert len(files) == 6
    shard = hf_hub_download(repo_id=REPO, repo_type='dataset', filename=files[0], revision=info.sha)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    parquet = pq.ParquetFile(shard)
    rows, seen = [], set()
    scanned = 0
    for batch in parquet.iter_batches(batch_size=256, columns=['conversation_id', 'conversation', 'language']):
        for record in batch.to_pylist():
            scanned += 1
            conversation_id = record['conversation_id']
            if not conversation_id or conversation_id in seen:
                continue
            messages = record['conversation'] or []
            if not isinstance(messages, list):
                continue
            # Use the longest valid context that ends at a user turn, at most
            # one prompt per conversation. The physical input is its last128.
            prefix = []
            chosen = None
            for message in messages:
                role = message.get('role') if isinstance(message, dict) else None
                content = message.get('content') if isinstance(message, dict) else None
                if role not in ('user', 'assistant') or not isinstance(content, str):
                    break
                prefix.append(dict(role=role, content=content))
                if role == 'user' and content.strip():
                    chosen = list(prefix)
            if chosen is None:
                continue
            try:
                ids = tokenizer.apply_chat_template(chosen, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False)
            except (ValueError, TypeError):
                continue
            if len(ids) < 128:
                continue
            seen.add(conversation_id)
            ids = ids[-128:]
            rows.append(dict(request_id=len(rows), source_row=scanned - 1,
                             conversation_id=conversation_id, language=record['language'],
                             input_ids=ids, input_ids_sha256=hashlib.sha256(bytes(json.dumps(ids), 'utf-8')).hexdigest(),
                             input_tokens=128))
            if len(rows) == 1024:
                break
        if len(rows) == 1024:
            break
    assert len(rows) == 1024, f'only {len(rows)} eligible distinct prompts in first official shard'
    output = ROOT / 'LMSYS_Chat_1M_requests.json'
    write(output, dict(status='PASS', dataset='LMSYS-Chat-1M', official_repo=REPO,
                       revision=info.sha, shard=files[0], shard_sha256=sha(shard),
                       model=str(MODEL), selection='First 1024 distinct conversations in official shard with a rendered user-ending Qwen prompt >=128 tokens; retain last128.',
                       scanned=scanned, pool_size=len(rows), requests=rows))
    print(output, flush=True)


if __name__ == '__main__':
    main()
