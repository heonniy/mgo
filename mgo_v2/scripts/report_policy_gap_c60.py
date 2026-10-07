"""Summarize clean policy timing and separate MoE diagnostics for completed cases."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
PACKET = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')
POLICIES = ('BR', 'CA_NATIVE', 'LA_CA_NEAR')
EP_COMPUTE_PHASES = frozenset(('moe.expert_compute',))
EP_COMM_PHASES = frozenset(('moe.forward_a2a', 'forward_token_a2a_submit',
                            'moe.forward_complete', 'moe.return_a2a',
                            'return_token_a2a'))


def analyze(label, expected_capacities=None, expected_p2p=None):
    path = ROOT / label
    result = json.loads((path / 'result.json').read_text())
    assert result['status'] == 'PASS' and result['policy_modes'] == list(POLICIES)
    assert result['prefetch'] == 'off' and result['decode_steps'] == 32
    assert result['main_capacities'] == (expected_capacities or [920, 920, 919, 919])
    if expected_p2p is not None:
        assert result['nccl_p2p_disable'] == ('1' if expected_p2p else None)
        for rank in range(4):
            source = json.loads((path / f'source_rank{rank}.json').read_text())
            assert source['nccl_p2p_disable'] == ('1' if expected_p2p else None)
    rows = {}
    for run, policy in enumerate(POLICIES):
        primary = json.loads((path / f'run{run}.json').read_text())
        assert primary['backend'] == policy and primary['status'] == 'PASS'
        assert primary['prefetch_issued'] == 0
        moe = np.zeros((4, 32, 48), dtype=np.float64)
        attention = np.zeros((4, 32, 48), dtype=np.float64)
        ep_compute = np.zeros((4, 32, 48), dtype=np.float64)
        ep_comm = np.zeros((4, 32, 48), dtype=np.float64)
        rank_rows = []
        diagnostics = []
        for rank in range(4):
            diag = json.loads((path / f'diagnostic_{policy}_rank{rank}.json').read_text())
            assert diag['token_prefix_parity'] and diag['decode_steps'] == 32
            diagnostics.append(diag)
            for segment in diag['segments']:
                event_index = segment['event_index']
                if 48 <= event_index < 48 * 33:
                    step, layer = divmod(event_index - 48, 48)
                    target = attention if segment['phase'] == 'attention_dense_residual' else moe
                    target[rank, step, layer] += segment['stream_seconds']
                    if segment['phase'] in EP_COMPUTE_PHASES:
                        ep_compute[rank, step, layer] += segment['stream_seconds']
                    elif segment['phase'] in EP_COMM_PHASES:
                        ep_comm[rank, step, layer] += segment['stream_seconds']
            assert np.all(moe[rank] > 0)
            rank_rows.append(dict(rank=rank, moe_seconds_per_decode=float(moe[rank].sum() / 32),
                                  cache_events=len(diag['decode_cache_events'])))
        moe_by_rank_step = moe.sum(axis=2)
        attention_by_rank_step = attention.sum(axis=2)
        total_by_rank_step = moe_by_rank_step + attention_by_rank_step
        critical_rank_by_step = np.argmax(total_by_rank_step, axis=0)
        steps = np.arange(32)
        critical_moe = float(moe_by_rank_step[critical_rank_by_step, steps].mean())
        critical_attention = float(attention_by_rank_step[critical_rank_by_step, steps].mean())
        critical_ep_compute = float(ep_compute.sum(axis=2)[critical_rank_by_step, steps].mean())
        critical_ep_comm = float(ep_comm.sum(axis=2)[critical_rank_by_step, steps].mean())
        critical_ep = critical_ep_compute + critical_ep_comm
        diagnostic_phase_total = float(total_by_rank_step[critical_rank_by_step, steps].mean())
        diagnostic_total = (max(d['output']['end_ns'] for d in diagnostics)
                            - max(d['output']['first_ns'] for d in diagnostics)) / 1e9 / 32
        assert abs(diagnostic_phase_total - diagnostic_total) <= 0.01 * diagnostic_total
        assert critical_moe <= diagnostic_phase_total and critical_moe <= diagnostic_total * 1.01
        assert 0 < critical_ep <= critical_moe
        rows[policy] = dict(primary_tpot_s=primary['TPOT'],
                            diagnostic_total_tpot_s=diagnostic_total,
                            diagnostic_phase_total_tpot_s=diagnostic_phase_total,
                            diagnostic_nonattention_tpot_s=critical_moe,
                            diagnostic_attention_tpot_s=critical_attention,
                            diagnostic_nonattention_share_pct=100 * critical_moe / diagnostic_phase_total,
                            diagnostic_ep_tpot_s=critical_ep,
                            diagnostic_ep_compute_tpot_s=critical_ep_compute,
                            diagnostic_ep_comm_tpot_s=critical_ep_comm,
                            diagnostic_ep_share_pct=100 * critical_ep / diagnostic_phase_total,
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
    if args.dataset == 'ShareGPT':
        lines = ['# ShareGPT C60 policy-gap physical results', '',
                 'R4 on GPUs 0,1,4,5; C60 MAIN3678 plus eight reserved P2 slots; native expert execution, compiled metadata, full pinned CPU source, prefetch OFF. Each case has 128 input tokens, 32 decode forwards, and one clean primary per policy after warmup. Routes and teacher inputs are frozen across BR, CA_NATIVE, and Near.', '',
                 '| B/rank | Seed sample/order | Policy | Clean TPOT s/token | Paired diagnostic total s/token | EP span s/token | Expert compute s/token | Expert comm s/token | Peer GiB | H2D GiB |',
                 '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for case in cases:
            spec = specs[case['cell']]
            seed = f"{spec['seed']['sample_seed']}/{spec['seed']['rank_order_seed']}"
            for policy in POLICIES:
                row = case['policies'][policy]
                lines.append(f"| {spec['local_batch']} | {seed} | {policy} | {row['primary_tpot_s']:.6f} | {row['diagnostic_total_tpot_s']:.6f} | {row['diagnostic_ep_tpot_s']:.6f} | {row['diagnostic_ep_compute_tpot_s']:.6f} | {row['diagnostic_ep_comm_tpot_s']:.6f} | {row['decode_peer_gib']:.3f} | {row['decode_h2d_gib']:.3f} |")
        lines += ['', 'Best observed clean TPOT gain versus the paired BR among the two measured physical seeds per batch (negative means slower):', '',
                  '| B/rank | CA_NATIVE seed / gain | Near seed / gain |', '|---:|---|---|']
        for batch in (8, 16, 64):
            winner = winners[str(batch)]
            def describe(policy):
                pick = winner[policy]['best_candidate']
                case = next(c for c in cases if c['label'] == pick['label'])
                seed = specs[case['cell']]['seed']
                return f"{seed['sample_seed']}/{seed['rank_order_seed']} / {pick['observed_gain_pct']:+.2f}%"
            lines.append(f"| {batch} | {describe('CA_NATIVE')} | {describe('LA_CA_NEAR')} |")
        lines += ['', 'Clean TPOT includes attention and is the performance result. Diagnostic total and EP span come from the same separate instrumented run. For each decode step, select the rank with the largest complete phase span. EP span is the exclusive sum of forward dispatch/finish, native expert execution, and return partial/combine scopes on that rank; controller, routing metadata, attention, and explicit H2D scopes are excluded. These current-stream spans can still contain host submission gaps, peer waits, and implicit H2D dependencies, so they are not pure NCCL/kernel service times. Instrumentation can slow the run; never subtract diagnostic EP from clean TPOT. One clean sample per policy does not establish repeatability.', '']
        (Path(__file__).resolve().parents[1] / 'experiments' / 'policy_gap_c60_20261008' / 'SHAREGPT_RESULTS.md').write_text('\n'.join(lines))
    print('|Batch|Seed|Policy|Clean TPOT s|Diagnostic total TPOT s|EP span s/token|Expert compute s/token|Expert comm s/token|Peer GiB|H2D GiB|')
    print('|---:|---|---|---:|---:|---:|---:|---:|---:|')
    for case in cases:
        cell = specs[case['cell']]
        for policy in POLICIES:
            p = case['policies'][policy]
            print(f"|{cell['local_batch']}|s{cell['seed']['sample_seed']}/d{cell['seed']['rank_order_seed']}|{policy}|{p['primary_tpot_s']:.6f}|{p['diagnostic_total_tpot_s']:.6f}|{p['diagnostic_ep_tpot_s']:.6f}|{p['diagnostic_ep_compute_tpot_s']:.6f}|{p['diagnostic_ep_comm_tpot_s']:.6f}|{p['decode_peer_gib']:.3f}|{p['decode_h2d_gib']:.3f}|")
    print(destination)
    print('Winners are among measured candidates only; one clean primary per policy.')


if __name__ == '__main__':
    main()
