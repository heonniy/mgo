"""Compare grouped and Ready-First policies on one identical quiet-host route."""

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
JOBS = {
    'new_OURS grouped': ROOT / 'grouped_frozen_policy_c30_b8_s14_d5_env2_quiet_v1',
    'main_OURS Ready-First': ROOT / 'main_ours_c30_b8_s14_d5_env2_quiet_pair_v1',
}
POLICIES = {'BR': 'BR', 'LA_CA_NEAR': 'Near', 'STATIC_MOD': 'Static'}


def main():
    cells = {}
    for path_name, path in JOBS.items():
        result = json.loads((path / 'result.json').read_text())
        assert result['status'] == 'PASS' and result['route_frozen']
        assert result['decode_steps'] == 32 and result['prefetch'] == 'off'
        assert result['nccl_p2p_disable'] == '1'
        policies = {}
        for backend, display in POLICIES.items():
            rows = [row for row in result['results'] if row['backend'] == backend]
            assert len(rows) == 2
            values = [row['TPOT'] for row in rows]
            policies[display] = {'samples': values, 'mean': statistics.mean(values),
                                 'min': min(values), 'max': max(values),
                                 'peer_bytes': rows[0]['decode_peer_bytes'],
                                 'h2d_bytes': rows[0]['decode_h2d_bytes']}
        br = policies['BR']['mean']; near = policies['Near']['mean']
        cells[path_name] = {'path': str(path), 'policies': policies,
                            'near_gain_vs_br_pct': 100 * (br - near) / br}
    route_match = []
    for rank in range(4):
        receipts = [json.loads((path / f'trace_rank{rank}.json').read_text())
                    for path in JOBS.values()]
        assert receipts[0]['route_sha256'] == receipts[1]['route_sha256']
        assert receipts[0]['teacher_tokens'] == receipts[1]['teacher_tokens']
        route_match.append(receipts[0]['route_sha256'])
    output = {'status': 'PASS', 'cell': 'ShareGPT_R4_C30_B8_L128_O33_s14_d5',
              'physical_gpus': [0, 1, 4, 5], 'quiet_gpus': [2, 3, 6, 7],
              'transport': 'p2p_disabled', 'trace_sha256_per_rank': route_match,
              'same_route_and_teacher': True, 'cells': cells}
    (HERE / 'QUIET_PAIR.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Quiet-host paired frozen-route policy check', '',
             'The four managed model-inference loads on GPUs 2/3/6/7 were '
             'stopped before these jobs; those GPUs were idle. Both jobs used '
             'R4 GPUs 0/1/4/5, ShareGPT input128, B8 per rank, 32 decode '
             'forwards, C30, P2P-disabled same-host transport, prefetch OFF, '
             'and the **same route hash and teacher tokens on all four ranks**. '
             'Each policy has two unfiltered cache-reset timings. Full TPOT '
             'includes attention.', '',
             '| Executor | BR TPOT (s/token) | Near TPOT (s/token) | Static TPOT (s/token) | Near vs BR |',
             '|---|---:|---:|---:|---:|']
    for name, cell in cells.items():
        def fmt(policy):
            row = cell['policies'][policy]
            return f"{row['mean']:.4f} [{row['min']:.4f}, {row['max']:.4f}]"
        lines.append(f"| {name} | {fmt('BR')} | {fmt('Near')} | {fmt('Static')} | "
                     f"{cell['near_gain_vs_br_pct']:+.2f}% |")
    lines += ['', 'The `main_OURS` Ready-First Near range is wholly below its BR '
              'range, whereas grouped `new_OURS` Near is wholly above BR. '
              'The executor change therefore reverses the observed policy '
              'ordering on this one frozen cell. It does not establish the '
              'same ordering for other batches, caches, or transports. '
              'Per-run timings, peer/H2D bytes, and raw paths are in '
              '[QUIET_PAIR.json](QUIET_PAIR.json).', '']
    (HERE / 'QUIET_PAIR.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
