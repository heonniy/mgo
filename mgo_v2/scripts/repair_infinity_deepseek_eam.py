"""Idempotently repair the pinned MoE-Infinity DeepSeek EAM wrapper.

The baseline environment is external to this repository. This script and its
hash receipt make the one-file environment repair reproducible and auditable.
"""
import hashlib
import json
from pathlib import Path


WRAPPERS = Path('/home/hwlee/mgo-tools/headline-r4/infinity-env/lib/python3.12/site-packages/moe_store/wrappers')
TARGETS = (WRAPPERS / 'deepseek.py', WRAPPERS / 'deepseek_v2_wrapper.py')
RECEIPT = Path(__file__).resolve().parents[1] / 'experiments/main_table_2x2_20261008/INFINITY_DEEPSEEK_EAM_REPAIR.json'
ANCHOR = '''        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states,
            routing_mask,
            routing_weight,
            router_logits=router_logits,
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()
'''
REPLACEMENT = '''        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states,
            routing_mask,
            routing_weight,
            router_logits=router_logits,
        )
        prefetcher = getattr(self, "expert_prefetcher", None)
        if getattr(prefetcher, "eam_enabled", False):
            selected = torch.topk(
                routing_weight, self.num_experts_per_tok, dim=-1
            ).indices.view(batch_size, sequence_length, self.num_experts_per_tok)
            selected = selected.detach().cpu().numpy()
            if len(self.seq_id_list) != batch_size:
                raise RuntimeError("DeepSeek EAM request identities do not match batch")
            scores = self.expert_predictor.predict_batch(
                self.seq_id_list, selected, self.layer_id
            )
            prefetcher.prefetch_eam(self.layer_id, scores)
        final_hidden_states = self.expert_executor.wait_dispatch_local()
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    files = []
    for target in TARGETS:
        before = target.read_bytes()
        source = before.decode()
        backup = target.with_suffix('.py.upstream')
        if REPLACEMENT in source:
            action = 'ALREADY_APPLIED'
        else:
            if source.count(ANCHOR) != 1:
                raise RuntimeError(f'unexpected DeepSeek wrapper source: {target}')
            if backup.exists() and backup.read_bytes() != before:
                raise RuntimeError(f'upstream backup differs from installed source: {target}')
            backup.write_bytes(before)
            target.write_text(source.replace(ANCHOR, REPLACEMENT))
            action = 'APPLIED'
        after = target.read_bytes()
        assert after.decode().count(REPLACEMENT) == 1
        files.append(dict(action=action, installed_file=str(target),
                          upstream_backup=str(backup),
                          upstream_sha256=sha(backup.read_bytes()),
                          patched_sha256=sha(after)))
    receipt = dict(status='PASS', files=files,
                   behavior='DeepSeek EAM predictor and budgeted prefetch called once per routed layer after dispatch submission, matching the repaired Qwen wrapper')
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'files': files}))


if __name__ == '__main__':
    main()
