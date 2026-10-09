"""Append independently measured Static TPOT to the prior main_OURS cells."""

import json
import statistics
from pathlib import Path

from run_static import GAP, OUTPUT, cases, check, label


HERE = Path(__file__).resolve().parent


def main():
    followup = {case['label']: case for case in json.loads((GAP / 'FOLLOWUP_RESULTS.json').read_text())['cases']}
    original = {case['label']: case for case in json.loads((GAP / 'SHAREGPT_RESULTS.json').read_text())['cases']}
    rows = []
    for cell, transport, old_label, _ in cases():
        run_label = label(cell, transport)
        attempt = 1
        while (OUTPUT / run_label).exists() and not (OUTPUT / run_label / 'result.json').exists():
            attempt += 1
            run_label = label(cell, transport).removesuffix('_v1') + f'_v{attempt}'
        check(old_label, run_label)
        previous = (followup | original)[old_label]
        result = json.loads((OUTPUT / run_label / 'result.json').read_text())
        samples = [run['TPOT'] for run in result['results']]
        assert len(samples) == 2
        rows.append({'cell': cell, 'transport': transport,
                     'cache_percent': int(cell.split('_')[2][1:]),
                     'batch_per_rank': int(cell.split('_')[3][1:]),
                     'seed': '/'.join(piece[1:] for piece in cell.split('_')[-2:]),
                     'previous_tpot_s': {policy: previous['policies'][policy]['primary_tpot_s']
                                         for policy in ('BR', 'CA_NATIVE', 'LA_CA_NEAR')},
                     'static_tpot_s': {'samples': samples, 'mean': statistics.mean(samples),
                                       'min': min(samples), 'max': max(samples)},
                     'previous_label': old_label, 'static_label': run_label,
                     'trace_and_teacher_match': True,
                     'raw_directory': str(OUTPUT / run_label)})
    (HERE / 'RESULTS.json').write_text(json.dumps({'status': 'PASS', 'cases': rows}, indent=2) + '\n')
    lines = ['# `main_OURS` fixed-owner Static placement', '',
             'R4 Qwen ShareGPT, input128, 32 decode forwards, prefetch OFF, '
             'native C++ Ready-First individual-expert execution. Static assigns '
             '`expert_id % 4` on GPUs 0/1/4/5. Each Static target starts after '
             'a cache reset and replays the exact original Near-captured route '
             'and teacher tokens; all four rank trace hashes and teacher tokens '
             'match the earlier cell. Static has **two unfiltered clean runs**; '
             'BR/CA/Near are the earlier **single** clean measurements, so their '
             'one-shot differences should not be treated as stable gains. '
             'The largest relative difference between the two Static repeats '
             'is 0.66%.', '',
             '| Transport | C | B/rank | Seed | Earlier BR | Earlier CA | Earlier Near | Added Static mean [range] (s/token) |',
             '|---|---:|---:|---|---:|---:|---:|---:|']
    for row in rows:
        p = row['previous_tpot_s']; s = row['static_tpot_s']
        lines.append(f"| {row['transport']} | {row['cache_percent']} | {row['batch_per_rank']} | "
                     f"{row['seed']} | {p['BR']:.4f} | {p['CA_NATIVE']:.4f} | "
                     f"{p['LA_CA_NEAR']:.4f} | {s['mean']:.4f} [{s['min']:.4f}, {s['max']:.4f}] |")
    lines += ['', 'All times are full TPOT including attention, in seconds per token. '
              'The original policy timings come from '
              '[SHAREGPT_RESULTS.json](../policy_gap_c60_20261008/SHAREGPT_RESULTS.json) '
              'and [FOLLOWUP_RESULTS.json](../policy_gap_c60_20261008/FOLLOWUP_RESULTS.json). '
              'Per-run Static values, provenance and raw receipt paths are in '
              '[RESULTS.json](RESULTS.json).', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
