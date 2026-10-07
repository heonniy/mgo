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
    output = dict(status='PASS', scope='12 selected-seed physical cases; each has one clean full-TPOT primary and separate paired EP diagnostic for BR/CA_NATIVE/Near',
                  cases=cases)
    destination = REPO / 'FOLLOWUP_RESULTS.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# C30 and P2P-disabled follow-up results', '',
             'ShareGPT, R4 GPUs0/1/4/5, input128,32 decode forwards, native main_OURS path, prefetch OFF, cold cache after warmup. The four unique seed cases were selected by the best observed C60/NVSwitch clean gain for each policy and batch; B8 has two seeds. Each row has one clean full-TPOT primary (attention included). Diagnostic values come from a separate instrumented run. P2P-disabled sets NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1 inside every worker to select SHM after the IB path failed; it is a transport setting on the same NVSwitch-equipped server.', '',
             '| Transport | C | B/rank | Seed sample/order | Policy | Clean TPOT ms/token | Diagnostic full decode s | First dispatch→last return s | EP layer windows s | Expert compute s | Dispatch/return s | Arrival/other s | Peer GiB | H2D GiB |',
             '|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in cases:
        seed = f"{case['seed']['sample_seed']}/{case['seed']['rank_order_seed']}"
        for policy in POLICIES:
            row = case['policies'][policy]
            lines.append(f"| {case['transport']} | {case['cache_percent']} | {case['local_batch']} | {seed} | {policy} | {row['primary_tpot_s']*1000:.3f} | {row['diagnostic_full_decode_s']:.3f} | {row['diagnostic_decode_dispatch_to_return_elapsed_s']:.3f} | {row['diagnostic_ep_critical_path_tpot_s']*32:.3f} | {row['diagnostic_ep_critical_compute_tpot_s']*32:.3f} | {row['diagnostic_ep_critical_comm_tpot_s']*32:.3f} | {row['diagnostic_ep_critical_other_tpot_s']*32:.3f} | {row['decode_peer_gib']:.3f} | {row['decode_h2d_gib']:.3f} |")
    lines += ['', '## Paired clean TPOT comparison', '',
              'Positive gain means faster than BR on the same seed. EP-window delta uses the separate diagnostic run and is not a clean-TPOT decomposition.', '',
              '| Transport | C | B/rank | Seed | BR TPOT ms/token | CA_NATIVE gain % | Near gain % | Near−BR EP windows s/32 steps |',
              '|---|---:|---:|---|---:|---:|---:|---:|']
    for case in cases:
        seed = f"{case['seed']['sample_seed']}/{case['seed']['rank_order_seed']}"
        br = case['policies']['BR']
        near = case['policies']['LA_CA_NEAR']
        ep_delta = (near['diagnostic_ep_critical_path_tpot_s'] - br['diagnostic_ep_critical_path_tpot_s']) * 32
        lines.append(f"| {case['transport']} | {case['cache_percent']} | {case['local_batch']} | {seed} | {br['primary_tpot_s']*1000:.3f} | {case['ca_gain_pct']:+.2f} | {case['near_gain_pct']:+.2f} | {ep_delta:+.3f} |")
    lines += ['', '## Interpretation', '']
    for transport, capacity in [('nvswitch', 30), ('p2p_disabled', 30), ('p2p_disabled', 60)]:
        group = [c for c in cases if c['transport'] == transport and c['cache_percent'] == capacity]
        near = [c['near_gain_pct'] for c in group]
        ca = [c['ca_gain_pct'] for c in group]
        peer_saved = [100 * (1 - c['policies']['CA_NATIVE']['decode_peer_gib'] /
                             c['policies']['BR']['decode_peer_gib']) for c in group]
        lines.append(f"- `{transport}` C{capacity}: Near observed clean gain {min(near):+.2f}% to {max(near):+.2f}%; CA_NATIVE {min(ca):+.2f}% to {max(ca):+.2f}% despite {min(peer_saved):.1f}%–{max(peer_saved):.1f}% fewer decode peer bytes than BR.")
    lines.append('- The C30 NVSwitch and P2P-disabled BR comparisons change sign across matched seeds, so these one-shot results do not establish a stable transport premium. P2P-disabled uses SHM on this same server; it is not a measurement on a physically non-NVLink server.')
    lines.append('- The EP-window sum is a bounded diagnostic of rank completion/communication behavior. It can diverge from the clean TPOT ordering because instrumentation and non-EP work affect the separate run; no pure communication-versus-expert causal attribution is claimed.')
    lines += ['', 'Policy effects must be computed within each same-seed, same-cache, same-transport group using clean TPOT. C60/NVSwitch reference rows are in SHAREGPT_RESULTS.md. All other timing columns are unnormalized seconds from one separate instrumented 32-step decode. “First dispatch→last return” is a single contiguous interval across the whole decode; it includes intervening attention/controller/H2D work and is not EP-only. “EP layer windows” sums the 48 nonoverlapping earliest-dispatch-to-latest-return intervals per decode step. Compute and dispatch/return are exclusive scopes on each layer’s last-return rank; arrival/other is remaining window time, including rank arrival skew and non-EP work after the first rank starts. Current-stream spans can include host gaps, peer waits, and implicit H2D dependencies; they are not pure kernel/network service. Never subtract these diagnostic values from clean TPOT. One timing sample per policy does not establish statistical stability.', '']
    (REPO / 'FOLLOWUP_RESULTS.md').write_text('\n'.join(lines))
    print(f'PASS {len(cases)} cases {destination}', flush=True)


if __name__ == '__main__':
    main()
