"""Audit a separate implementation candidate without pooling old timings."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from adaptive_timing import paired_decision
from refactor_fingerprint import assert_equivalent


def summarize(root, stage='R4_H64_NATIVE_V3', staging_backend='memmove', unique_combine=False, async_metadata_inputs=False):
    batches, identities, sources = {}, {}, {}
    for batch in (128, 256):
        folder = root / f'{stage}_B{batch}_H64'
        data = {}
        for name in ('result.json', 'status.json'):
            path = folder / name
            raw = path.read_bytes()
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
            data[name] = json.loads(raw)
        result, state = data['result.json'], data['status.json']
        assert state['status'] == result['status'] == 'PASS'
        group = state['group']
        assert (group['world'], group['gpus'], group['batch'], group['horizon']) == (4, [0, 1, 4, 5], batch, 64)
        assert result['baseline'] == 'BR' and result['candidate'] == 'LA'
        assert result['physical_arena_count'] == 1
        cases = result['cases']
        assert cases == group['cases']
        assert [c['policy'] for c in cases] == ['BR', 'LA']
        for case in cases:
            assert case.get('unique_combine', False) == unique_combine
            assert case.get('async_metadata_inputs', False) == async_metadata_inputs
            for key, value in dict(runtime_arm='V3_OPT_PF_OVERLAP', staging_backend=staging_backend, horizon=64, partial_precision='bf16', P=2, trigger='T2', overlap=True).items():
                assert case[key] == value, (key, case)
        gate = paired_decision(result['pairs'])
        assert gate['complete'] and 2 <= len(result['pairs']) <= 3
        if unique_combine or async_metadata_inputs:
            for repeat in range(1, len(result['pairs']) + 1):
                for policy in ('BR', 'LA'):
                    receipts = []
                    for rank in range(4):
                        path = folder / f'{policy}_r{repeat}_measure_rank{rank}.json'
                        raw = path.read_bytes()
                        receipt = json.loads(raw)
                        assert receipt['status'] == 'PASS' and receipt['rank'] == rank
                        if unique_combine:
                            assert receipt['unique_combine_layers'] == 64 * 48
                            assert receipt['case']['unique_combine'] is True
                        if async_metadata_inputs:
                            assert receipt['async_metadata_inputs'] is True
                            assert receipt['case']['async_metadata_inputs'] is True
                        sources[str(path)] = hashlib.sha256(raw).hexdigest()
                        receipts.append(receipt)
                    for metric in ('E2E_wall', 'TPOT'):
                        assert max(r[metric] for r in receipts) == result['pairs'][repeat - 1][policy][metric]
        stats = {}
        for policy in ('BR', 'LA'):
            stats[policy] = {}
            for metric in ('E2E_wall', 'TPOT'):
                values = [p[policy][metric] for p in result['pairs']]
                stats[policy][metric] = dict(mean=statistics.mean(values), median=statistics.median(values), full_range=[min(values), max(values)])
        batches[str(batch)] = dict(gate=gate, statistics=stats, samples=result['pairs'])
        identities[('native', batch)] = state['common_stack']
    assert_equivalent(identities)
    stable = all(not b['gate']['unstable'] for b in batches.values())
    positive = stable and all(b['gate']['gain']['TPOT']['positive_supported'] for b in batches.values())
    return dict(status='STABLE_CANDIDATE' if stable else 'EXCLUDED_UNSTABLE', improvement_supported=positive,
                world=4, physical_gpus=[0, 1, 4, 5], horizon=64, precision='bf16', staging_backend=staging_backend, unique_combine=unique_combine, async_metadata_inputs=async_metadata_inputs,
                robust_TPOT_gain=min(b['gate']['gain']['TPOT']['estimate'] for b in batches.values()),
                batches=batches, source_hashes=sources,
                scope='Separate implementation candidate; all pairs retained; no pooling with other implementations. This report does not activate a default or complete phase/CA validation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', default='R4_H64_NATIVE_V3')
    parser.add_argument('--staging-backend', choices=['torch', 'memmove'], default='memmove')
    parser.add_argument('--unique-combine', action='store_true')
    parser.add_argument('--async-metadata-inputs', action='store_true')
    args = parser.parse_args()
    report = summarize(args.root, args.stage, args.staging_backend, args.unique_combine, args.async_metadata_inputs)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(report['status'])
