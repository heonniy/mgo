"""Summarize clean policy timing and separate MoE diagnostics for completed cases."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
WORKLOADS = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008/WORKLOADS.json')
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
            for event in diag['moe_events']:
                if 1 <= event['step'] <= 32:
                    moe[rank, event['step'] - 1, event['layer']] = event['stream_seconds']
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
    args = parser.parse_args()
    cases = [analyze(x) for x in args.labels]
    destination = WORKLOADS.parent / 'SHAREGPT_RESULTS.json'
    destination.write_text(json.dumps(dict(status='PASS', cases=cases), indent=2) + '\n')
    print('|Batch|Seed|Policy|Clean TPOT s|Diagnostic MoE s/token|Peer GiB|H2D GiB|')
    print('|---:|---|---|---:|---:|---:|---:|')
    for case in cases:
        cell = next(c for c in json.loads(WORKLOADS.read_text())['cells'] if c['cell'] == case['cell'])
        for policy in POLICIES:
            p = case['policies'][policy]
            print(f"|{cell['local_batch']}|s{cell['seed']['sample_seed']}/d{cell['seed']['rank_order_seed']}|{policy}|{p['primary_tpot_s']:.6f}|{p['diagnostic_moe_tpot_s']:.6f}|{p['decode_peer_gib']:.3f}|{p['decode_h2d_gib']:.3f}|")
    print(destination)


if __name__ == '__main__':
    main()
