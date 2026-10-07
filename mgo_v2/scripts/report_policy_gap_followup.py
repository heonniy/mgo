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
                if transport == 'nvswitch' and capacity == 30 and spec['local_batch'] == 8 and spec['case_id'] == 0:
                    label = 'policy_gap_sharegpt_c30_nvswitch_b8_case0_v2_20261008'
                status = json.loads((ROOT / label / 'status.json').read_text())
                assert status['status'] == 'PASS' and status['nccl_p2p_disable'] == (transport == 'p2p_disabled')
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
             'ShareGPT, R4 GPUs0/1/4/5, input128,32 decode forwards, native main_OURS path, prefetch OFF, cold cache after warmup. The four unique seed cases were selected by the best observed C60/NVSwitch clean gain for each policy and batch; B8 has two seeds. Each row has one clean primary. The MoE value comes from a separate instrumented run and excludes attention. P2P-disabled sets NCCL_P2P_DISABLE=1 inside every worker; it is a transport setting on the same NVSwitch-equipped server.', '',
             '| Transport | C | B/rank | Seed sample/order | Policy | Clean TPOT s/token | MoE-block TPOT s/token (diagnostic) | Peer GiB | H2D GiB |',
             '|---|---:|---:|---|---|---:|---:|---:|---:|']
    for case in cases:
        seed = f"{case['seed']['sample_seed']}/{case['seed']['rank_order_seed']}"
        for policy in POLICIES:
            row = case['policies'][policy]
            lines.append(f"| {case['transport']} | {case['cache_percent']} | {case['local_batch']} | {seed} | {policy} | {row['primary_tpot_s']:.6f} | {row['diagnostic_moe_tpot_s']:.6f} | {row['decode_peer_gib']:.3f} | {row['decode_h2d_gib']:.3f} |")
    lines += ['', 'Policy effects must be computed within each same-seed, same-cache, same-transport group. C60/NVSwitch reference rows are in SHAREGPT_RESULTS.md. The diagnostic uses current-stream MLP spans including host gaps and peer waits, so it is not pure GPU compute and must not be subtracted from clean TPOT. One timing sample per policy does not establish statistical stability.', '']
    (REPO / 'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines))
    print(f'PASS {len(cases)} cases {destination}', flush=True)


if __name__ == '__main__':
    main()
