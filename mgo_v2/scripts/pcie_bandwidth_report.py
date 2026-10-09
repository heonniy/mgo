"""Derive effective H2D bandwidth from retained PCIe microbenchmark rows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcie_pool_size_report import samples


GPUS = (0, 1, 4, 5)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(v):
    return dict(n=len(v), median=float(np.median(v)), mean=float(np.mean(v)),
                sd=float(np.std(v, ddof=1)), p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)))


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    runs = {'144MiB compact': a.small, '54GiB compact': a.compact, '54GiB spread': a.spread}
    output = dict(status='PASS', units='decimal GB/s (10^9 bytes/s)',
                  seed=20261009, bootstrap_draws=10000, runs={}, comparisons=[], cells=[])
    rng = np.random.default_rng(20261009)
    raw_out = []
    arrays = {}
    for label, root in runs.items():
        _, summary = samples(root)  # Verify 30 repeats, raw makespans, pinning, NUMA and shared inodes.
        rows = list(csv.DictReader((root / 'microbench_raw.csv').open()))
        output['runs'][label] = dict(root=str(root), source_bytes_per_node=summary.get('source_bytes_per_node', 144 * 2**20),
            sha256={n: digest(root / n) for n in ('microbench_raw.csv', 'microbench_summary.json') + tuple(f'host_rank{r}.json' for r in range(4))})
        for cell in summary['cells']:
            name = cell['name']
            by_repeat = [[r for r in rows if r['cell'] == name and int(r['repeat']) == i] for i in range(30)]
            for scope, ranks in [('system', range(4)), ('switch_0_1', (0, 1)), ('switch_4_5', (2, 3))] + [(f'GPU{g}_completion', (r,)) for r, g in enumerate(GPUS)]:
                if not sum(cell['counts'][r] for r in ranks):
                    continue
                v = []
                for i, block in enumerate(by_repeat):
                    active = [r for r in block if int(r['rank']) in ranks and int(r['bytes']) > 0]
                    byte_count = sum(int(r['bytes']) for r in active)
                    ns = max(int(r['ready_ns']) for r in active) - int(block[0]['start_ns'])
                    assert ns > 0
                    bw = byte_count / ns  # bytes/ns equals decimal GB/s.
                    v.append(bw)
                    raw_out.append(dict(source=label, cell=name, repeat=i, scope=scope, bytes=byte_count,
                                        endpoint_ms=ns / 1e6, effective_GBps=bw))
                arrays[label, name, scope] = np.array(v)
                if scope.startswith('switch'):
                    np.testing.assert_allclose(np.median(v), cell['group_GBps'][0 if scope == 'switch_0_1' else 1]['median'])
                output['cells'].append(dict(source=label, cell=name, scope=scope, streams=cell['streams'], remote=cell['remote'], overlap=cell['overlap'], **statistics(v)))
            # max(per-stream event duration) is not a shared multi-stream window.
            # CUDA service bandwidth is therefore defined only for single-stream cells.
            if cell['streams'] != 1:
                continue
            for rank, gpu in enumerate(GPUS):
                if not cell['counts'][rank]:
                    continue
                v = []
                for i, block in enumerate(by_repeat):
                    row = next(r for r in block if int(r['rank']) == rank)
                    ms = float(row['service_ms']); byte_count = int(row['bytes'])
                    assert ms > 0 and byte_count == cell['counts'][rank] * 9437184
                    bw = byte_count / (ms * 1e6)
                    v.append(bw)
                    raw_out.append(dict(source=label, cell=name, repeat=i, scope=f'GPU{gpu}_CUDA_service', bytes=byte_count,
                                        endpoint_ms=ms, effective_GBps=bw))
                np.testing.assert_allclose(np.median([float(next(r for r in b if int(r['rank']) == rank)['service_ms']) for b in by_repeat]), cell['ranks'][rank]['service_ms']['median'])
                arrays[label, name, f'GPU{gpu}_CUDA_service'] = np.array(v)
                output['cells'].append(dict(source=label, cell=name, scope=f'GPU{gpu}_CUDA_service', streams=1, remote=cell['remote'], overlap=cell['overlap'], **statistics(v)))
        for rank, gpu in enumerate(GPUS):
            pair = 'two_pair_0_1' if rank < 2 else 'two_pair_2_3'
            cross = 'two_pair_0_2' if rank in (0, 2) else 'two_pair_1_3'
            key = f'GPU{gpu}_CUDA_service'
            solo, same, independent = (arrays[label, n, key] for n in (f'two_single_{rank}', pair, cross))
            index = rng.integers(0, 30, (10000, 30))
            row = dict(source=label, gpu=gpu, solo_cell=f'two_single_{rank}', same_switch_cell=pair, cross_switch_cell=cross,
                       solo_GBps=float(np.median(solo)), same_switch_GBps=float(np.median(same)),
                       cross_switch_GBps=float(np.median(independent)))
            for ref, values in [('solo', solo), ('cross_switch', independent)]:
                row[f'drop_vs_{ref}_pct'] = float(100 * (1 - np.median(same) / np.median(values)))
                boot = 100 * (1 - np.median(same[index], axis=1) / np.median(values[index], axis=1))
                row[f'drop_vs_{ref}_ci95_pct'] = np.percentile(boot, [2.5, 97.5]).tolist()
            output['comparisons'].append(row)

    (a.out / 'bandwidth.json').write_text(json.dumps(output, indent=2) + '\n')
    for filename, values in [('bandwidth_comparisons.csv', output['comparisons']), ('bandwidth_cells.csv', output['cells']), ('bandwidth_samples.csv', raw_out)]:
        with (a.out / filename).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    lines = ['# Effective PCIe H2D bandwidth', '',
             'Primary evidence: two physical 54-GiB NUMA-local shared pinned sources, spread addresses. GPUs 0/1 share one PCIe group; GPUs 4/5 share the other. Existing measurements are converted; no new GPU timing run was performed.', '',
             '**Units and endpoints.** Decimal GB/s = bytes / seconds / 1e9. Per-GPU CUDA service bandwidth uses the recorded start/end CUDA events on its single copy stream. Switch and system bandwidth use the common CPU release to the last active GPU ready in that scope, including host scheduling and launch/completion overhead. Switch aggregate is bytes actually transferred on that switch divided by that window; it is not the sum of independent per-GPU service rates.', '',
             '**Payload control.** Each fetch is 9,437,184 bytes (9 MiB). Solo transfers are two sequential fetches on one GPU (18 MiB); same-switch transfers are one fetch on each of two GPUs (9 MiB/GPU, 18 MiB/switch). The solo comparison therefore has a per-GPU payload-count difference. Cross-switch controls use one 9-MiB fetch per GPU, on separate switches, and give the matched-per-GPU-payload comparison. This is synthetic BF16 H2D, not measured generation throughput or a PCIe hardware link-rate claim.', '',
             '| Pinned source / GPU | Solo service GB/s | Same-switch service GB/s | Drop vs solo | Cross-switch service GB/s | Drop vs matched cross-switch |',
             '|---|---:|---:|---:|---:|---:|']
    for c in output['comparisons']:
        lo, hi = c['drop_vs_solo_ci95_pct']; cl, ch = c['drop_vs_cross_switch_ci95_pct']
        lines.append(f"| {c['source']} / {c['gpu']} | {c['solo_GBps']:.3f} | {c['same_switch_GBps']:.3f} | {c['drop_vs_solo_pct']:.2f}% [{lo:.2f}, {hi:.2f}] | {c['cross_switch_GBps']:.3f} | {c['drop_vs_cross_switch_pct']:.2f}% [{cl:.2f}, {ch:.2f}] |")
    lines += ['', 'Values are medians of per-repeat bandwidth. Brackets are 95% paired repeat-block bootstrap intervals for the percentage drop (10,000 draws, seed 20261009), pairing conditions only within the same job. All 30 timed repeats after five warmups are included. No cross-job bootstrap pairing or causal pool-size claim is made.', '',
              '| Pinned source / switch | Solo first GPU GB/s | Solo second GPU GB/s | Both on same switch GB/s | Two separate switches: system GB/s |',
              '|---|---:|---:|---:|---:|']
    for label in runs:
        for node, pair in enumerate(('two_pair_0_1', 'two_pair_2_3')):
            scope = ('switch_0_1', 'switch_4_5')[node]
            med = lambda name, s=scope: np.median(arrays[label, name, s])
            lines.append(f"| {label} / {scope} | {med(f'two_single_{node*2}'):.3f} | {med(f'two_single_{node*2+1}'):.3f} | {med(pair):.3f} | {med('two_pair_0_2', 'system'):.3f} |")
    lines += ['', '| Pinned source | Six-fetch R 4:2 system GB/s | Six-fetch G 3:3 system GB/s | Effective bandwidth increase |', '|---|---:|---:|---:|']
    for label in runs:
        r = np.median(arrays[label, 'six_0', 'system']); g = np.median(arrays[label, 'six_1', 'system'])
        lines.append(f'| {label} | {r:.3f} | {g:.3f} | {100*(g/r-1):.2f}% |')
    lines += ['', 'Six-fetch values divide the identical 56,623,104-byte payload by the common-release-to-last-ready window. Bandwidth increase is G/R minus one; it differs numerically from latency reduction (one minus R/G bandwidth). These are isolated H2D results, not generation TPOT improvements.', '',
              'The same-switch per-GPU loss with approximately preserved switch aggregate is consistent with sharing a group bottleneck. The matched cross-switch controls support this interpretation despite the solo payload-count difference. These measurements do not isolate whether the limiting shared resource is the switch upstream link, root port, or another shared path component.', '',
              'The CSV/JSON artifacts also convert every original microbench condition, including six-fetch placement, scaling, remote-source and overlap sensitivity. Those controls remain explicitly labelled. Multi-stream cells have completion-window bandwidth only: their recorded maximum individual stream duration does not define a valid common CUDA service window.', '',
              'Pinning, physical sharing, sampled NUMA page locality, repeat counts and original summary timing/bandwidth agreement are validated against the retained raw rows and host receipts. Input SHA256 digests are in bandwidth.json. See [the pinned pool comparison](../microbench_pool_comparison/RESULTS.md) for full source allocation and layout details.', '',
              'Reproduce from repository root:', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data2/esjung/envs/mgo-pcie/bin/python mgo_v2/scripts/pcie_bandwidth_report.py \\',
              f'  --small {a.small} --compact {a.compact} --spread {a.spread} --out /tmp/pcie-bandwidth-report',
              '```', '']
    (a.out / 'RESULTS.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    primary = [c for c in output['comparisons'] if c['source'] == '54GiB spread']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    x = np.arange(4)
    for i, (key, label, color) in enumerate([('solo_GBps', 'Solo: 18 MiB/GPU', '#2563A6'), ('same_switch_GBps', 'Same switch: 9 MiB/GPU', '#B7791F'), ('cross_switch_GBps', 'Separate switches: 9 MiB/GPU', '#4A7A39')]):
        axes[0].bar(x + (i-1)*.25, [c[key] for c in primary], .25, label=label, color=color)
    axes[0].set_xticks(x, [f'GPU {g}' for g in GPUS]); axes[0].set(ylabel='Per-GPU CUDA service (GB/s)', ylim=(0, 15)); axes[0].legend(fontsize=8)
    for i, (name, label, color) in enumerate([('first', 'Solo first GPU', '#2563A6'), ('second', 'Solo second GPU', '#4A7A39'), ('pair', 'Both GPUs on same switch', '#B7791F')]):
        values = []
        for node, pair in enumerate(('two_pair_0_1', 'two_pair_2_3')):
            cell = pair if name == 'pair' else f'two_single_{node*2+(name=="second")}'
            values.append(np.median(arrays['54GiB spread', cell, ('switch_0_1', 'switch_4_5')[node]]))
        axes[1].bar(np.arange(2) + (i-1)*.25, values, .25, label=label, color=color)
    axes[1].set_xticks([0, 1], ['Switch: GPUs 0/1', 'Switch: GPUs 4/5']); axes[1].set(ylabel='Switch aggregate, release to ready (GB/s)', ylim=(0, 15)); axes[1].legend(fontsize=8)
    fig.suptitle('Effective H2D bandwidth: 54 GiB shared pinned per NUMA node\nSpread addresses; medians of 30 repeats; single copy stream per GPU')
    fig.savefig(a.out / 'bandwidth.pdf'); fig.savefig(a.out / 'preview.png', dpi=150); plt.close(fig)
    print(json.dumps(dict(status='PASS', primary=primary)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('small', 'compact', 'spread', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
