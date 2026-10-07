"""Freeze policy-specific observed C60 seed winners for C30/env2 follow-up."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    source_path = ROOT / 'WORKLOADS.json'
    results_path = ROOT / 'SHAREGPT_RESULTS.json'
    source = json.loads(source_path.read_text())
    result = json.loads(results_path.read_text())
    assert result['status'] == 'PASS' and len(result['cases']) == 6
    by_cell = {cell['cell']: cell for cell in source['cells']}
    by_label = {case['label']: case for case in result['cases']}
    selections = []
    unique = []
    for batch in (8, 16, 64):
        for policy in ('CA_NATIVE', 'LA_CA_NEAR'):
            choice = result['bounded_observed_winners'][str(batch)][policy]['best_candidate']
            case = by_label[choice['label']]
            cell = by_cell[case['cell']]
            assert cell['local_batch'] == batch
            selections.append(dict(batch=batch, policy=policy, source_cell=cell['cell'],
                                   label=choice['label'], signed_c60_nv_gain_pct=choice['observed_gain_pct']))
            if cell['cell'] not in [x['cell'] for x in unique]:
                unique.append(cell)
    assert len(unique) == 4
    cells = []
    for capacity in (30, 60):
        for original in unique:
            cell = copy.deepcopy(original)
            cell['cell'] = original['cell'].replace('_C60_', f'_C{capacity}_')
            cell['expert_slots_per_rank'] = [461, 461, 461, 460] if capacity == 30 else [922, 922, 921, 921]
            cell['capacity_percent'] = capacity
            cell['source_cell'] = original['cell']
            cells.append(cell)
    output = ROOT / 'FOLLOWUP_WORKLOADS.json'
    write(output, dict(status='FROZEN', dataset='ShareGPT', input_tokens=128, decode_forwards=32,
                       source_workloads=str(source_path), source_workloads_sha256=sha(source_path),
                       source_results=str(results_path), source_results_sha256=sha(results_path),
                       selection='Per-policy best observed clean TPOT gain versus BR among two physically measured C60/NVSwitch seed candidates per batch; union of winning seeds.',
                       selections=selections, unique_seed_cases=len(unique), cells=cells))
    print(f'PASS {output}: {len(unique)} unique seed cases, {len(cells)} capacity cells', flush=True)


if __name__ == '__main__':
    main()
