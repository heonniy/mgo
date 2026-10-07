"""Summarize clean policy timing and separate MoE diagnostics for completed cases."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
PACKET = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')
POLICIES = ('BR', 'CA_NATIVE', 'LA_CA_NEAR')


def analyze(label):
    path = ROOT / label
    result = json.loads((path / 'result.json').read_text())
    assert result['status'] == 'PASS' and result['policy_modes'] == list(POLICIES)
    assert result['prefetch'] == 'off' and result['decode_steps'] == 32
    assert result['main_capacities'] == [920, 920, 919, 919]
    rows = {}
    for run, policy in enumerate(POLICIES):
        primary = json.loads((path / f'run{run}.json').read_text())
        assert primary['backend'] == policy and primary['status'] == 'PASS'
        assert primary['prefetch_issued'] == 0
        moe = np.zeros((4, 32, 48), dtype=np.float64)
        rank_rows = []
        for rank in range(4):
            diag = json.loads((path / f'diagnostic_{policy}_rank{rank}.json').read_text())
            assert diag['token_prefix_parity'] and diag['decode_steps'] == 32
            for segment in diag['segments']:
                event_index = segment['event_index']
                if 48 <= event_index < 48 * 33 and segment['phase'] != 'attention_dense_residual':
                    step, layer = divmod(event_index - 48, 48)
                    moe[rank, step, layer] += segment['stream_seconds']
            assert np.all(moe[rank] > 0)
            rank_rows.append(dict(rank=rank, moe_seconds_per_decode=float(moe[rank].sum() / 32),
                                  cache_events=len(diag['decode_cache_events'])))
        rows[policy] = dict(primary_tpot_s=primary['TPOT'],
                            diagnostic_moe_tpot_s=float(moe.sum(axis=2).max(axis=0).mean()),
                            decode_peer_gib=primary['decode_peer_bytes'] / 2**30,
                            decode_h2d_gib=primary['decode_h2d_bytes'] / 2**30,
                            token_agreement_to_br=primary['token_agreement_to_reference'],
                            ranks=rank_rows)
    return dict(status='PASS', label=label, cell=result['cell'], policies=rows,
                ca_gain_pct=100 * (1 - rows['CA_NATIVE']['primary_tpot_s'] / rows['BR']['primary_tpot_s']),
                near_gain_pct=100 * (1 - rows['LA_CA_NEAR']['primary_tpot_s'] / rows['BR']['primary_tpot_s']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--dataset', choices=('ShareGPT', 'LMSYS'), default='ShareGPT')
    args = parser.parse_args()
    workloads = PACKET / ('WORKLOADS.json' if args.dataset == 'ShareGPT' else 'LMSYS_WORKLOADS.json')
    specs = {cell['cell']: cell for cell in json.loads(workloads.read_text())['cells']}
    cases = [analyze(x) for x in args.labels]
    winners = {}
    for batch in (8, 16, 64):
        subset = [case for case in cases if specs[case['cell']]['local_batch'] == batch]
        if subset:
            winners[str(batch)] = {}
            for policy, key in [('CA_NATIVE', 'ca_gain_pct'), ('LA_CA_NEAR', 'near_gain_pct')]:
                best = max(subset, key=lambda case: case[key])
                largest = max(subset, key=lambda case: abs(case[key]))
                winners[str(batch)][policy] = dict(best_candidate=dict(label=best['label'], observed_gain_pct=best[key]),
                                                   largest_absolute_gap=dict(label=largest['label'], signed_gain_pct=largest[key]))
    destination = PACKET / f'{args.dataset.upper()}_RESULTS.json'
    destination.write_text(json.dumps(dict(status='PASS', dataset=args.dataset, cases=cases,
                                           bounded_observed_winners=winners), indent=2) + '\n')
    print('|Batch|Seed|Policy|Clean TPOT s|Diagnostic MoE s/token|Peer GiB|H2D GiB|')
    print('|---:|---|---|---:|---:|---:|---:|')
    for case in cases:
        cell = specs[case['cell']]
        for policy in POLICIES:
            p = case['policies'][policy]
            print(f"|{cell['local_batch']}|s{cell['seed']['sample_seed']}/d{cell['seed']['rank_order_seed']}|{policy}|{p['primary_tpot_s']:.6f}|{p['diagnostic_moe_tpot_s']:.6f}|{p['decode_peer_gib']:.3f}|{p['decode_h2d_gib']:.3f}|")
    print(destination)
    print('Winners are among measured candidates only; one clean primary per policy.')


if __name__ == '__main__':
    main()
