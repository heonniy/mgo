"""Bound DeepSeek speculative EAM transfers while retaining all priority scores.

Only workers that set eam_max_speculative_per_gpu use the limit. Existing Qwen
baseline behavior is unchanged. The GPU expert residency budget remains C30.
"""
import hashlib
import json
from pathlib import Path


TARGET = Path('/home/hwlee/mgo-tools/headline-r4/MoE-Infinity/moe_infinity/memory/expert_prefetcher.py')
RECEIPT = Path(__file__).resolve().parents[1] / 'experiments/main_table_2x2_20261008/INFINITY_EAM_BACKPRESSURE.json'
ANCHOR = '''        for score, tensor_id in candidates:
            gpu = self.eam_homes[tensor_id]
            if used[gpu] + self.eam_expert_bytes > self.eam_budget[gpu]:
                continue
            used[gpu] += self.eam_expert_bytes
            admitted.append((tensor_id, gpu))
'''
REPLACEMENT = '''        speculative_limit = getattr(self, "eam_max_speculative_per_gpu", None)
        if speculative_limit is not None and speculative_limit < 1:
            raise ValueError("EAM speculative limit must be positive")
        for score, tensor_id in candidates:
            gpu = self.eam_homes[tensor_id]
            if used[gpu] + self.eam_expert_bytes > self.eam_budget[gpu]:
                continue
            if (speculative_limit is not None and
                    used[gpu] // self.eam_expert_bytes >= speculative_limit):
                continue
            used[gpu] += self.eam_expert_bytes
            admitted.append((tensor_id, gpu))
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    before = TARGET.read_bytes()
    source = before.decode()
    backup = TARGET.with_suffix('.py.upstream')
    if REPLACEMENT in source:
        action = 'ALREADY_APPLIED'
    else:
        if source.count(ANCHOR) != 1:
            raise RuntimeError('unexpected EAM admission source')
        if backup.exists() and backup.read_bytes() != before:
            raise RuntimeError('EAM upstream backup differs from installed source')
        backup.write_bytes(before)
        TARGET.write_text(source.replace(ANCHOR, REPLACEMENT))
        action = 'APPLIED'
    after = TARGET.read_bytes()
    assert after.decode().count(REPLACEMENT) == 1
    receipt = dict(status='PASS',action=action,installed_file=str(TARGET),
                   upstream_backup=str(backup),upstream_sha256=sha(backup.read_bytes()),
                   patched_sha256=sha(after),
                   behavior='Only workers setting eam_max_speculative_per_gpu cap speculative enqueues; full ranked priorities and C30 byte budgets remain in force; Qwen unchanged')
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:receipt[k] for k in ('status','action','patched_sha256')}))


if __name__ == '__main__':
    main()
