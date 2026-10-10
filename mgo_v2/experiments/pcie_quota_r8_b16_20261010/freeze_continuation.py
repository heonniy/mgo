"""Freeze the Near target's generated tokens as common future model inputs."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs/near_quota_full_v1')


def main():
    status = json.loads((SOURCE / 'status.json').read_text())
    assert status['status'] == 'PASS' and status['physical_gpus'] == list(range(8))
    rank_tokens = {}
    request_ids = {}
    for rank in range(8):
        row = json.loads((SOURCE / f'repeat1_rank{rank}.json').read_text())
        assert row['policy'] == 'LA_CA_NEAR' and row['phase'] == 'target'
        assert row['expert_cache_start'] == 'empty' and row['output_tokens'] == 64
        assert len(row['tokens']) == len(row['request_ids']) == 16
        assert all(len(tokens) == 64 for tokens in row['tokens'])
        rank_tokens[str(rank)] = row['tokens']
        request_ids[str(rank)] = row['request_ids']
    data = dict(status='FROZEN', cell='Qwen3_ShareGPT_R8_C30_B16_L512_O64',
                local_batch=16, output_tokens=64,
                source_job=str(SOURCE), source_commit=status['source_commit'],
                workload_sha256=status['workload_sha256'],
                rank_tokens=rank_tokens, request_ids=request_ids)
    target = HERE / 'FROZEN_TOKENS.json'
    target.write_text(json.dumps(data, indent=2) + '\n')
    print(hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
