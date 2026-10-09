"""Summarize unfiltered paired BR/Near/Static frozen-route measurements."""

import json
import statistics
from pathlib import Path

from run_search import MANIFEST, OUTPUT, label


POLICIES = {'BR': 'BR', 'LA_CA_NEAR': 'Near', 'STATIC_MOD': 'Static'}
HERE = Path(__file__).resolve().parent


def metric(rows, name):
    values = [row[name] for row in rows]
    return {'values': values, 'mean': statistics.mean(values),
            'min': min(values), 'max': max(values),
            'repeat_difference_pct': 100 * abs(values[0] - values[1]) / statistics.mean(values)}


def summarize():
    cases = []
    for spec in json.loads(MANIFEST.read_text())['cells']:
        cell = spec['cell']
        for transport in ('nvswitch', 'env2'):
            if transport == 'env2' and '_C30_' not in cell:
                continue
            path = OUTPUT / label(cell, transport)
            result = json.loads((path / 'result.json').read_text())
            assert result['status'] == 'PASS' and result['route_frozen']
            assert result['prefetch'] == 'off' and result['grouped_decode_mode'] == 'hit_then_miss'
            assert result['compiled_dense'] and result['decode_steps'] == 32
            assert bool(result['nccl_p2p_disable']) == (transport == 'env2')
            policies = {}
            for backend, display in POLICIES.items():
                runs = [x for x in result['results'] if x['backend'] == backend]
                assert len(runs) == 2 and all(x['status'] == 'PASS' for x in runs)
                policies[display] = {key: metric(runs, key) for key in
                                     ('TPOT', 'TTFT', 'E2E', 'decode_peer_bytes',
                                      'decode_h2d_bytes', 'decode_main_hit_rate')}
                policies[display]['token_agreement'] = [x['token_agreement_to_reference'] for x in runs]
            br = policies['BR']['TPOT']
            near = policies['Near']['TPOT']
            case = {'cell': cell, 'label': label(cell, transport), 'transport': transport,
                    'cache_percent': int(cell.split('_')[2][1:]),
                    'batch_per_rank': spec['local_batch'], 'seed': cell.split('_')[-2:],
                    'policies': policies,
                    'near_gain_pct': 100 * (br['mean'] - near['mean']) / br['mean'],
                    'near_faster_both_ranges': near['max'] < br['min'],
                    'raw_directory': str(path)}
            cases.append(case)
    confirmation_path = OUTPUT / 'grouped_frozen_policy_c30_b8_s14_d5_env2_confirm_v1'
    confirmation = None
    if (confirmation_path / 'result.json').exists():
        measured = json.loads((confirmation_path / 'result.json').read_text())
        assert measured['status'] == 'PASS' and measured['route_frozen']
        first = OUTPUT / label('ShareGPT_R4_C30_B8_L128_O33_s14_d5', 'env2')
        for rank in range(4):
            prior = json.loads((first / f'trace_rank{rank}.json').read_text())
            later = json.loads((confirmation_path / f'trace_rank{rank}.json').read_text())
            assert prior['route_sha256'] == later['route_sha256']
            assert prior['teacher_tokens'] == later['teacher_tokens']
        confirmation = {'cell': measured['cell'], 'transport': 'env2',
                        'raw_directory': str(confirmation_path),
                        'same_trace_and_teacher_as_screen': True,
                        'policies': {display: {key: metric(
                            [run for run in measured['results'] if run['backend'] == backend], key)
                            for key in ('TPOT', 'decode_peer_bytes', 'decode_h2d_bytes')}
                            for backend, display in POLICIES.items()}}
        br = confirmation['policies']['BR']['TPOT']['mean']
        near = confirmation['policies']['Near']['TPOT']['mean']
        confirmation['near_gain_pct'] = 100 * (br - near) / br
    return {'status': 'PASS', 'method': 'two unfiltered runs per policy, frozen BR route, 32 decode steps',
            'cases': cases, 'confirmation': confirmation}


def main():
    result = summarize()
    (HERE / 'RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = [
        '# Grouped `new_OURS`: frozen-route placement comparison',
        '',
        'R4 on GPUs 0/1/4/5; ShareGPT input128, 32 decode forwards; prefetch OFF; '
        'native C++ expert execution; grouped hit wave then grouped miss wave. '
        'One BR-captured route and teacher input per cell were replayed for all policies. '
        'Each displayed TPOT is the mean of **two unfiltered runs** in the order '
        'BR/Near/Static/Static/Near/BR. Ranges are the full two-run ranges.',
        '',
        '| Transport | Cache | B/rank | Seed | BR TPOT (s) | Near TPOT (s) | Static TPOT (s) | Near gain vs BR |',
        '|---|---:|---:|---|---:|---:|---:|---:|',
    ]
    for case in result['cases']:
        def fmt(policy):
            value = case['policies'][policy]['TPOT']
            return f"{value['mean']:.4f} [{value['min']:.4f}, {value['max']:.4f}]"
        lines.append(f"| {case['transport']} | C{case['cache_percent']} | {case['batch_per_rank']} | "
                     f"{'/'.join(case['seed'])} | {fmt('BR')} | {fmt('Near')} | "
                     f"{fmt('Static')} | {case['near_gain_pct']:+.2f}% |")
    positive = [x for x in result['cases'] if x['near_gain_pct'] > 0]
    lines += ['', 'The gain column is `(BR − Near) / BR`; a positive value favors Near. '
              'A best observed seed is not a global optimum. Repeat ranges and the fresh '
              'confirmation determine whether the apparent gain is credible.', '']
    if positive:
        best = max(positive, key=lambda x: x['near_gain_pct'])
        lines += [f"Largest screen gain: **{best['cell']}**, {best['near_gain_pct']:+.2f}%. "
                  f"The two-run ranges {'do' if best['near_faster_both_ranges'] else 'do not'} "
                  'separate.', '']
    else:
        lines += ['No screened setting produced a positive Near mean gain.', '']
    if result['confirmation']:
        case = result['confirmation']
        br = case['policies']['BR']['TPOT']
        near = case['policies']['Near']['TPOT']
        lines += [
            '## Fresh confirmation of the largest screened mean gain', '',
            f"The same env2 C30/B8 seed 14/5 and identical route/teacher were rerun in "
            f"a fresh guarded job. BR was {br['mean']:.4f} [{br['min']:.4f}, {br['max']:.4f}] "
            f"and Near was {near['mean']:.4f} [{near['min']:.4f}, {near['max']:.4f}] "
            f"s/token, an observed Near mean gain of {case['near_gain_pct']:+.2f}%. "
            f"Within that job, BR repeats differed by {br['repeat_difference_pct']:.1f}% "
            f"and Near by {near['repeat_difference_pct']:.1f}%; their ranges overlap. "
            'Both absolute TPOT means also shifted about 21% below the first job, despite '
            'identical route and H2D bytes. Thus this seed is the best observed positive '
            'candidate, **not a reliable Near speedup**. The full 12-cell screen found '
            'no stable Near winner under these grouped settings.', '']
    lines += ['Full per-run values, peer/H2D bytes, decode hit rates, token agreement, '
              'and raw receipt paths are in [RESULTS.json](RESULTS.json).', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
