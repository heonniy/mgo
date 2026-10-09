"""Summarize all H2D GPU combinations and render concurrency figures."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1] / 'experiments/pcie_rank_concurrency_20261009'
GPUS = (0, 1, 3, 4, 5, 6, 7)


def main():
    result = json.loads((ROOT / 'RESULTS.json').read_text())
    assert result['status'] == 'PASS' and result['count'] == 254
    rows = result['rows']
    assert all(row['repeats'] == 2 for row in rows)
    by = {(tuple(row['gpus']), row['payload']): row for row in rows}
    solo = {(gpu, payload): by[(gpu,), payload]['rank_gib_per_s'][str(gpu)]['median']
            for gpu in GPUS for payload in ('Qwen expert', 'DeepSeek expert')}
    table = []
    for row in rows:
        for gpu in row['gpus']:
            speed = row['rank_gib_per_s'][str(gpu)]
            table.append(dict(payload=row['payload'], gpus='-'.join(map(str, row['gpus'])),
                              count=row['concurrent_ranks'], gpu=gpu,
                              rank_gib_per_s=speed['median'], rank_min=speed['minimum'],
                              rank_max=speed['maximum'], solo_gib_per_s=solo[gpu, row['payload']],
                              ratio_to_solo=speed['median'] / solo[gpu, row['payload']],
                              aggregate_gib_per_s=row['aggregate_gib_per_s']['median'],
                              start_skew_ms=row['max_start_skew_ms']))
    with (ROOT / 'RANK_COMBINATIONS.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(table)
    summaries = []
    for payload in ('Qwen expert', 'DeepSeek expert'):
        for count in range(1, len(GPUS) + 1):
            subsets = [row for row in rows if row['payload'] == payload and row['concurrent_ranks'] == count]
            ratios = [row['ratio_to_solo'] for row in table if row['payload'] == payload and row['count'] == count]
            speeds = [row['aggregate_gib_per_s']['median'] for row in subsets]
            summaries.append(dict(payload=payload, simultaneous_ranks=count,
                                  combinations=len(subsets), aggregate_gib_per_s_median=statistics.median(speeds),
                                  aggregate_gib_per_s_min=min(row['aggregate_gib_per_s']['minimum'] for row in subsets),
                                  aggregate_gib_per_s_max=max(row['aggregate_gib_per_s']['maximum'] for row in subsets),
                                  rank_ratio_to_solo_median=statistics.median(ratios),
                                  rank_ratio_to_solo_min=min(ratios), rank_ratio_to_solo_max=max(ratios)))
    with (ROOT / 'CARDINALITY.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(summaries)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True, layout='constrained')
    for i, payload in enumerate(('Qwen expert', 'DeepSeek expert')):
        groups = [[row['aggregate_gib_per_s']['median'] for row in rows
                   if row['payload'] == payload and row['concurrent_ranks'] == count]
                  for count in range(1, 8)]
        ratio_groups = [[row['ratio_to_solo'] for row in table
                         if row['payload'] == payload and row['count'] == count]
                        for count in range(1, 8)]
        for j, values in enumerate((groups, ratio_groups)):
            ax = axes[i, j]
            parts = ax.boxplot(values, positions=range(1, 8), widths=.56,
                               patch_artist=True, showfliers=True, whis=(0, 100))
            color = '#0072B2' if j == 0 else '#E69F00'
            for patch in parts['boxes']:
                patch.set(facecolor=color, alpha=.7, edgecolor='#263645')
            for median in parts['medians']:
                median.set(color='#1b2430', linewidth=1.4)
            ax.set_ylim(bottom=0)
            ax.set_title(f'{payload} · {"aggregate" if j == 0 else "rank-local"}')
            ax.grid(axis='y', color='#dce3e8', linewidth=.7)
            ax.spines[['top', 'right']].set_visible(False)
        axes[i, 0].set_ylabel('Aggregate effective H2D (GiB/s)')
        axes[i, 1].set_ylabel('Rank speed / same-rank solo speed')
    for ax in axes[1]:
        ax.set_xlabel('Simultaneous GPU count')
    fig.suptitle('Pinned-host H2D · all 127 subsets of GPUs 0,1,3,4,5,6,7', fontsize=12)
    fig.savefig(ROOT / 'pcie_concurrency.png', dpi=240, facecolor='white')
    fig.savefig(ROOT / 'pcie_concurrency.pdf', facecolor='white')
    plt.close(fig)
    lines = ['# PCIe host-to-device concurrency', '',
             'Pinned rank-private host source; 256 sequential asynchronous copies per repeat; '
             'two unfiltered repeats. Each GPU worker was bound to its own CPU core. '
             'This reports effective service under the measured host and GPU conditions, '
             'not PCIe theoretical line rate or model E2E speed. Ranges include all '
             'combinations of that cardinality and both repeats.', '',
             '| Payload | Active GPUs | Subsets | Aggregate GiB/s, median [min, max] | '
             'Rank speed / solo, median [min, max] |',
             '|---|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['payload']} | {row['simultaneous_ranks']} | {row['combinations']} | "
                     f"{row['aggregate_gib_per_s_median']:.2f} "
                     f"[{row['aggregate_gib_per_s_min']:.2f}, {row['aggregate_gib_per_s_max']:.2f}] | "
                     f"{row['rank_ratio_to_solo_median']:.3f} "
                     f"[{row['rank_ratio_to_solo_min']:.3f}, {row['rank_ratio_to_solo_max']:.3f}] |")
    max_skew = max(row['max_start_skew_ms'] for row in rows)
    lines.extend(['', '## Per-rank solo versus seven-way', '',
                  '| GPU | Qwen solo | Qwen seven-way | Qwen ratio | '
                  'DeepSeek solo | DeepSeek seven-way | DeepSeek ratio |',
                  '|---:|---:|---:|---:|---:|---:|---:|'])
    for gpu in GPUS:
        q0 = solo[gpu, 'Qwen expert']
        q7 = by[tuple(GPUS), 'Qwen expert']['rank_gib_per_s'][str(gpu)]['median']
        d0 = solo[gpu, 'DeepSeek expert']
        d7 = by[tuple(GPUS), 'DeepSeek expert']['rank_gib_per_s'][str(gpu)]['median']
        lines.append(f'| {gpu} | {q0:.2f} | {q7:.2f} | {q7 / q0:.3f} | '
                     f'{d0:.2f} | {d7:.2f} | {d7 / d0:.3f} |')
    lines.extend(['', 'All bandwidth columns use GiB/s.', '', '## Combination effect', ''])
    lines.append('| Payload | GPU 0+1+3 aggregate | GPU 0+1+4 aggregate | All 7 aggregate | '
                 'All 7 / sum of solos |')
    lines.append('|---|---:|---:|---:|---:|')
    for payload in ('Qwen expert', 'DeepSeek expert'):
        triple = by[(0, 1, 3), payload]['aggregate_gib_per_s']['median']
        control = by[(0, 1, 4), payload]['aggregate_gib_per_s']['median']
        all_seven = by[tuple(GPUS), payload]['aggregate_gib_per_s']['median']
        ideal = sum(solo[gpu, payload] for gpu in GPUS)
        lines.append(f'| {payload} | {triple:.2f} | {control:.2f} | '
                     f'{all_seven:.2f} | {100 * all_seven / ideal:.1f}% |')
    lines.extend(['', f'Maximum observed worker start skew: {max_skew:.3f} ms.',
                  'Individual rank and subset results, including full ranges, are in '
                  '`RANK_COMBINATIONS.csv` and `RESULTS.json`.',
                  'The 0+1+3 penalty reproduces at both payload sizes and both repeats. '
                  'This benchmark cannot isolate whether PCIe links, host memory, '
                  'IOMMU or the virtualized upstream fabric caused it.'])
    if max_skew > 5:
        lines.append('Some subsets had >5 ms start skew; inspect those rows before interpreting them as simultaneous.')
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
