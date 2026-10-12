"""Freeze NEAR (serial + inline H2D, live tokens) output IDs as the DeepSeek R8 teacher continuation."""
import hashlib, json
from pathlib import Path

ROOT = Path('/home/hwlee/mgo-results/deepseek_r8_sharegpt_b16_l512_20261012')
JOB = ROOT / 'jobs/dsteacher_full_v1'
CELL = 'DeepSeekV2Lite_ShareGPT_R8_C30_B16_L512_O64'
status = json.loads((JOB / 'status.json').read_text())
assert status['status'] == 'PASS' and status['decode_policy'] == 'LA_CA_NEAR'
ranks = [json.loads((JOB / f'repeat1_rank{r}.json').read_text()) for r in range(8)]
assert all(x['phase'] == 'target' and not x['forced_continuation'] for x in ranks)
out = dict(status='FROZEN', cell=CELL, local_batch=16, output_tokens=64, source_job=str(JOB),
           source_commit=status['source_commit'], source_dirty=status['source_dirty'],
           workload_sha256=status['workload_sha256'],
           rank_tokens={str(r): x['tokens'] for r, x in enumerate(ranks)},
           request_ids={str(r): x['request_ids'] for r, x in enumerate(ranks)})
assert all(len(t) == 16 and all(len(row) == 64 for row in t) for t in out['rank_tokens'].values())
path = Path(__file__).resolve().parent / 'FROZEN_TOKENS.json'
path.write_text(json.dumps(out) + '\n')
print('frozen', path, hashlib.sha256(path.read_bytes()).hexdigest())
