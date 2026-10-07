"""Aggregate selected-seed C30/NVSwitch and C30/C60 P2P-disabled receipts."""
import json
from pathlib import Path

from report_policy_gap_c60 import PACKET, ROOT, POLICIES, analyze

MANIFEST = PACKET / 'FOLLOWUP_WORKLOADS.json'
REPO = Path(__file__).resolve().parents[1] / 'experiments' / 'policy_gap_c60_20261008'


def main():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['status'] == 'FROZEN' and len(manifest['cells']) == 8
    cases = []
    for transport, capacities in [('nvswitch', (30,)), ('p2p_disabled', (30, 60))]:
        for capacity in capacities:
            for spec in manifest['cells']:
                if spec['capacity_percent'] != capacity:
                    continue
                label = f"policy_gap_sharegpt_c{capacity}_{transport}_b{spec['local_batch']}_case{spec['case_id']}_20261008"
                if transport == 'nvswitch' and capacity == 30 and spec['local_batch'] == 8:
                    label = f"policy_gap_sharegpt_c30_nvswitch_b8_case{spec['case_id']}_v2_20261008"
                if transport == 'p2p_disabled' and capacity == 30 and spec['local_batch'] == 8 and spec['case_id'] == 0:
                    label = 'policy_gap_sharegpt_c30_p2p_disabled_b8_case0_v2_20261008'
                status = json.loads((ROOT / label / 'status.json').read_text())
                assert status['status'] == 'PASS' and status['nccl_p2p_disable'] == (transport == 'p2p_disabled')
                assert status.get('nccl_ib_disable', False) == (transport == 'p2p_disabled')
                for rank in range(4):
                    source = json.loads((ROOT / label / f'source_rank{rank}.json').read_text())
                    assert source.get('nccl_ib_disable') == ('1' if transport == 'p2p_disabled' else None)
                expected = [459, 459, 459, 458] if capacity == 30 else [920, 920, 919, 919]
                result = analyze(label, expected, transport == 'p2p_disabled')
                result.update(cache_percent=capacity, transport=transport,
                              local_batch=spec['local_batch'], seed=spec['seed'], source_cell=spec['source_cell'])
                cases.append(result)
    assert len(cases) == 12
    output = dict(status='PASS', scope='12 selected-seed physical cases; each has one clean primary and separate MoE diagnostic for BR/CA_NATIVE/Near',
                  cases=cases)
    destination = REPO / 'FOLLOWUP_RESULTS.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# C30 and P2P-disabled follow-up results', '',
             'ShareGPT, R4 GPUs0/1/4/5, input128,32 decode forwards, native main_OURS path, prefetch OFF, cold cache after warmup. The four unique seed cases were selected by the best observed C60/NVSwitch clean gain for each policy and batch; B8 has two seeds. Each row has one clean full-TPOT primary (attention included). The EP value comes from a separate instrumented run and includes only expert execution and dispatch/return scopes. P2P-disabled sets NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1 inside every worker to select SHM after the IB path failed; it is a transport setting on the same NVSwitch-equipped server.', '',
             '| Transport | C | B/rank | Seed sample/order | Policy | Clean TPOT s/token | Paired diagnostic total s/token | EP span s/token | Expert compute s/token | Expert comm s/token | Peer GiB | H2D GiB |',
             '|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        seed = f"{case['seed']['sample_seed']}/{case['seed']['rank_order_seed']}"
        for policy in POLICIES:
            row = case['policies'][policy]
            lines.append(f"| {case['transport']} | {case['cache_percent']} | {case['local_batch']} | {seed} | {policy} | {row['primary_tpot_s']:.6f} | {row['diagnostic_total_tpot_s']:.6f} | {row['diagnostic_ep_tpot_s']:.6f} | {row['diagnostic_ep_compute_tpot_s']:.6f} | {row['diagnostic_ep_comm_tpot_s']:.6f} | {row['decode_peer_gib']:.3f} | {row['decode_h2d_gib']:.3f} |")
    lines += ['', 'Policy effects must be computed within each same-seed, same-cache, same-transport group using clean TPOT. C60/NVSwitch reference rows are in SHAREGPT_RESULTS.md. Diagnostic total and EP span come from the same instrumented run: select each decode step\'s slowest rank using its complete phase span, then take that rank\'s forward dispatch/finish, expert execution, and return partial/combine scopes. Controller, routing metadata, attention, and explicit H2D scopes are excluded. Current-stream spans can still include host gaps, peer waits, and implicit H2D dependencies; they are not pure kernel or network service. Never subtract EP span from clean TPOT. One timing sample per policy does not establish statistical stability.', '']
    (REPO / 'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines))
    print(f'PASS {len(cases)} cases {destination}', flush=True)


if __name__ == '__main__':
    main()
