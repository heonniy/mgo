"""Reproduce the standalone PDF and paired confidence intervals from raw rows."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args()
    summary=json.loads((a.root/'microbench_summary.json').read_text());cells={c['name']:c for c in summary['cells']}
    source_note=f"{summary.get('source_bytes_per_node',144*2**20)/2**30:g} GiB pinned/node; {summary.get('source_layout','compact')} addresses"
    raw=list(csv.DictReader((a.root/'microbench_raw.csv').open()))
    samples={}
    for name in cells:
        values=[]
        for repeat in range(30):
            rows=[r for r in raw if r['cell']==name and int(r['repeat'])==repeat]
            assert len(rows)==4 and len({r['start_ns'] for r in rows})==1
            active=[r for r in rows if int(r['bytes'])]
            values.append((max(int(r['ready_ns']) for r in active)-int(rows[0]['start_ns']))/1e6)
        samples[name]=np.asarray(values)
        np.testing.assert_allclose(np.median(values),cells[name]['makespan_ms']['median'])
    rng=np.random.default_rng(20261009);comparisons=[]
    for rank,group in [('six_0','six_1'),('six_0','six_2')]+[(f'scale_{m}_rank',f'scale_{m}_group') for m in (2,4,6,8,10,14,50,62)]:
        index=rng.integers(0,30,(10000,30))
        boot=np.median(samples[rank][index],axis=1)/np.median(samples[group][index],axis=1)
        ratio=float(np.median(samples[rank])/np.median(samples[group]))
        comparisons.append(dict(rank=rank,group=group,median_speedup_ratio=ratio,paired_bootstrap_ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                                latency_reduction_fraction=1-1/ratio))
    (a.root/'comparisons.json').write_text(json.dumps(dict(seed=20261009,bootstrap_draws=10000,method='paired repeat-block bootstrap of ratio of medians',comparisons=comparisons),indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'axes.axisbelow':True})
    blue='#2563A6';gold='#B7791F'
    with PdfPages(a.root/'microbench_plot.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        names=['two_single_0','two_pair_0_1','two_pair_0_2','two_pair_2_3','two_single_2']
        labels=['GPU0 ×2','GPU0+1','GPU0+4','GPU4+5','GPU4 ×2']
        axes[0].boxplot([samples[n] for n in names],tick_labels=labels,showfliers=True,medianprops={'color':blue},widths=.6)
        axes[0].set(title='Two 9-MiB fetches',ylabel='Common start to last ready (ms)',ylim=(0,None))
        names=['six_0','six_1','six_2','six_3'];labels=['2,2,1,1','1,2,1,2','2,1,2,1','1,1,2,2']
        axes[1].boxplot([samples[n] for n in names],tick_labels=labels,showfliers=True,medianprops={'color':blue},widths=.6)
        axes[1].set(title='Six fetches; physical GPU order 0,1,4,5',ylabel='Common start to last ready (ms)',ylim=(0,None))
        fig.suptitle('Isolated H2D, verified local NUMA pinned sources — '+source_note,fontsize=13)
        fig.text(.5,-.02,'30 samples per cell after 5 warmups; randomized repeat blocks. Boxes: quartiles; lines: medians; dots: outliers.',ha='center',fontsize=9)
        pdf.savefig(fig,bbox_inches='tight');fig.savefig(a.root/'microbench_preview.png',dpi=150,bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        counts=[2,4,6,8,10,14,50,62]
        for mode,color,marker in [('rank',blue,'o'),('group',gold,'s')]:
            names=[f'scale_{m}_{mode}' for m in counts]
            med=np.array([np.median(samples[n]) for n in names]);ci=np.array([cells[n]['median_ci95_ms'] for n in names])
            axes[0].errorbar(counts,med,yerr=np.abs(np.stack([med-ci[:,0],ci[:,1]-med])),color=color,marker=marker,label=mode,capsize=3)
        axes[0].set(title='Miss-count scaling',xlabel='Global fetch count M',ylabel='Makespan median (ms)',ylim=(0,None));axes[0].legend()
        ratios=comparisons[2:];v=np.array([r['median_speedup_ratio'] for r in ratios]);ci=np.array([r['paired_bootstrap_ci95'] for r in ratios])
        axes[1].errorbar(range(8),v,yerr=np.abs(np.stack([v-ci[:,0],ci[:,1]-v])),fmt='o',color=blue,capsize=3)
        axes[1].axhline(1,color='#444444',linestyle='--');axes[1].set_xticks(range(8),counts)
        axes[1].set(title='Rank/group ratio of medians',xlabel='Global fetch count M',ylabel='Speedup ratio; paired bootstrap 95% CI')
        fig.suptitle('One copy stream per GPU; M=4 and M=8 are negative controls\n'+source_note,fontsize=13)
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        for name,color,marker in [('rank',blue,'o'),('group',gold,'s')]:
            names=[f'scale_14_{name}',f'concurrency_2_{name}',f'concurrency_4_{name}']
            axes[0].plot([1,2,4],[np.median(samples[n]) for n in names],color=color,marker=marker,label=name)
        axes[0].set(title='M=14: per-GPU copy concurrency',xlabel='Copy streams per GPU',ylabel='Makespan median (ms)',ylim=(0,None));axes[0].set_xticks([1,2,4]);axes[0].legend()
        names=['six_0','six_1','remote_0','remote_1','overlap_0','overlap_1']
        labels=['Local R','Local G','Remote R','Remote G','G2G+R','G2G+G']
        axes[1].boxplot([samples[n] for n in names],tick_labels=labels,medianprops={'color':blue},widths=.6)
        axes[1].set(title='M=6: source/overlap sensitivity',ylabel='H2D last-ready median (ms)',ylim=(0,None))
        fig.suptitle('Remote pages verified; G2G overlap is a separate diagnostic\n'+source_note,fontsize=13)
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    text=['# Physical PCIe microbenchmark — 2026-10-09','',
          f"Pinned source per NUMA node: {summary.get('source_bytes_per_node',144*2**20)/2**30:g} GiB. Address layout: {summary.get('source_layout','compact')}. Unique physical source: {2*summary.get('source_bytes_per_node',144*2**20)/2**30:g} GiB.", '',
          'Completed on physical GPUs 0,1,4,5. One 9,437,184-byte BF16 payload per fetch; one shared pinned source per NUMA node. '
          'All local/remote mappings have sampled page-location receipts. Every isolated sample ran before NCCL was initialized. '
          'Forty-one cells, five warmups and thirty timed samples per cell; all warmup and timed rows retained.','',
          '| Comparison | Rank/skew median (ms) | Group median (ms) | Ratio | Paired 95% CI |','|---|---:|---:|---:|---:|']
    for c in comparisons:
        lo,hi=c['paired_bootstrap_ci95'];text.append(f"| {c['rank']} / {c['group']} | {np.median(samples[c['rank']]):.4f} | {np.median(samples[c['group']]):.4f} | {c['median_speedup_ratio']:.4f} | {lo:.4f}–{hi:.4f} |")
    text+=['','The ratio describes isolated copy makespan, not model TPOT. Thirty randomized paired repeat blocks support a 10,000-draw '
           'bootstrap interval for the ratio of medians. Per-rank CUDA event service and host completion, group bandwidth, '
           'p10/p90/p99 and all raw rows are separate artifacts. Group bandwidth is decimal GB/s from common release to that group’s last active rank.',
           '', 'Directional peer costs use measured two-rank NCCL broadcast service, 9-MiB-minus-4-KiB median transfer slopes, normalized to integer scale 1000. '
           'This calibration contains only sender and receiver; it is not a full all-to-all latency model. Across groups NCCL selected SHM; within groups P2P/CUMEM. '
           'Those transport differences must remain visible in the topology-weighted policy interpretation.',
           '', 'Earlier G2_microbench_attempt1 failed before timing because CUDA host registration had no current context. G2_microbench_attempt2 completed samples but failed in '
           'PyTorch 2.5 P2P communicator teardown under per-rank single-GPU visibility. G2_microbench_attempt3 replaced calibration P2P calls with two-rank '
           'collectives, completed all measurements, unregistered sources, destroyed groups, and exited cleanly. Those earlier attempts remain under the raw output root; they are not failures of this confirmation job.',
           '',f'Raw root: `{a.root}`. No model-generation performance is claimed by this experiment.','']
    (a.root/'MICROBENCH_RESULTS.md').write_text('\n'.join(text))


if __name__=='__main__':main()
