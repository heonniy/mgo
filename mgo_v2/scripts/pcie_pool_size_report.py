"""Compare 144-MiB and full 54-GiB pinned sources from independent runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pcie_host import write


def samples(root):
    raw=list(csv.DictReader((root/'microbench_raw.csv').open()))
    summary=json.loads((root/'microbench_summary.json').read_text())
    values={}
    for cell in summary['cells']:
        name=cell['name'];timings=[]
        for repeat in range(30):
            rows=[r for r in raw if r['cell']==name and int(r['repeat'])==repeat]
            assert len(rows)==4 and len({r['start_ns'] for r in rows})==1
            assert cell['overlap'] or all(r['nccl_initialized']=='False' for r in rows)
            active=[r for r in rows if int(r['bytes'])]
            timings.append((max(int(r['ready_ns']) for r in active)-int(rows[0]['start_ns']))/1e6)
        values[name]=np.array(timings)
        np.testing.assert_allclose(np.median(timings),cell['makespan_ms']['median'])
    receipts=[json.loads((root/f'host_rank{r}.json').read_text()) for r in range(4)]
    for node in (0,1):
        inode=set()
        for rank,receipt in enumerate(receipts):
            source=next(x for x in receipt['sources'] if x['numa_node']==node)
            inode.add((source['device'],source['inode']))
            if rank//2==node:
                assert source['pinned']
                assert set(source['locality']['node_counts'])=={str(node)}
        assert len(inode)==1
    return values,summary


def main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    runs={'144MiB compact':a.small,'54GiB compact':a.compact,'54GiB spread':a.spread}
    loaded={label:samples(root) for label,root in runs.items()}
    rng=np.random.default_rng(20261009);rows=[]
    names=list(loaded['144MiB compact'][0])
    for name in names:
        ref=loaded['144MiB compact'][0][name]
        for label,(values,summary) in loaded.items():
            v=values[name]
            # Different jobs are independent; never pair their sample indexes.
            left=rng.integers(0,30,(10000,30));right=rng.integers(0,30,(10000,30))
            boot=np.median(v[left],axis=1)/np.median(ref[right],axis=1)
            rows.append(dict(cell=name,source=label,median_ms=float(np.median(v)),p10_ms=float(np.percentile(v,10)),p90_ms=float(np.percentile(v,90)),p99_ms=float(np.percentile(v,99)),
                             ratio_to_144MiB=float(np.median(v)/np.median(ref)),ratio_ci95=np.percentile(boot,[2.5,97.5]).tolist()))
    for label in ('54GiB compact','54GiB spread'):
        assert loaded[label][1]['source_bytes_per_node']==54*2**30
    comparisons={}
    for label,(v,summary) in loaded.items():
        index=rng.integers(0,30,(10000,30))
        ratio=np.median(v['six_0'])/np.median(v['six_1'])
        bootstrap=np.median(v['six_0'][index],axis=1)/np.median(v['six_1'][index],axis=1)
        comparisons[label]=dict(rank_ms=float(np.median(v['six_0'])),group_ms=float(np.median(v['six_1'])),rank_group_ratio=float(ratio),
                               latency_reduction_fraction=float(1-1/ratio),paired_ratio_ci95=np.percentile(bootstrap,[2.5,97.5]).tolist())
    write(a.out/'pool_comparison.json',dict(status='PASS',runs={k:str(v) for k,v in runs.items()},cells=rows,six_fetch=comparisons,
        notes=['Full pools use the same SharedPinnedMapping/cudaHostRegister path as the model runtime; one 54-GiB source physically shared by each GPU pair.',
               'Main full-size matrix registers only each rank\'s local 54-GiB pool. Remote registration happens later, outside main timings.',
               'Compact reads the initial 16 expert rows (144 MiB), isolating pinned footprint; spread changes addresses across all 6144 expert rows with equal-M arms sharing global payload IDs.',
               'Full-size remote controls follow the local matrix; the older compact run interleaved remote cells. Cross-job ratios use independent bootstrap and may include temporal drift.',
               'Synthetic BF16 payloads, not model weights. Physical source size and copy path match the real expert store; this is not full-generation latency.',
               'Original peer weights remain frozen; new calibration results are retained as a sensitivity measurement.']))
    with (a.out/'pool_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# NUMA shared pinned pool size comparison','',
          '| Source per NUMA | R 4:2 (ms) | G 3:3 (ms) | R/G | H2D latency reduction | Paired 95% CI |',
          '|---|---:|---:|---:|---:|---:|']
    for label,c in comparisons.items():
        lo,hi=c['paired_ratio_ci95'];text.append(f"| {label} | {c['rank_ms']:.4f} | {c['group_ms']:.4f} | {c['rank_group_ratio']:.4f} | {c['latency_reduction_fraction']*100:.2f}% | {lo:.4f}–{hi:.4f} |")
    text+=['','Each full-size run has two physically unique 54-GiB pools (108 GiB total), shared by GPUs0/1 and4/5 respectively. CUDA pinning, same inodes, local page placement, correctness warmups and unregister receipts are checked.',
           '', 'Compact keeps a 144-MiB active source region; spread rotates payload locations throughout the 54-GiB pool. All41 conditions retain30timed samples after5warmups. The complete CSV reports every cell, p10/p90/p99 and independent-job uncertainty.',
           '', 'Between-job differences include temporal drift; they do not establish a causal model speedup. The within-job R/G intervals use paired repeat-block bootstrap. Remote controls in full-size jobs run after the local matrix, registering the other node outside main timings. Both local pools are already physically resident and pinned by their own GPU pairs.',
           '', 'The synthetic transfer path matches full pinned source allocation and registration, but contents are synthetic. Model generation is reported separately. The original frozen G-NUMA-CA peer calibration is retained.', '']
    (a.out/'RESULTS.md').write_text('\n'.join(text))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    colors=['#2563A6','#B7791F','#4A7A39']
    with PdfPages(a.out/'pool_comparison.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        for i,(label,(v,_)) in enumerate(loaded.items()):
            med=np.array([np.median(v['six_0']),np.median(v['six_1'])])
            idx=rng.integers(0,30,(10000,30))
            ci=np.array([np.percentile(np.median(v[n][idx],axis=1),[2.5,97.5]) for n in ('six_0','six_1')])
            axes[0].errorbar([0,1],med,yerr=np.abs(np.stack([med-ci[:,0],ci[:,1]-med])),marker='o',color=colors[i],label=label,capsize=3)
            counts=[2,4,6,8,10,14,50,62]
            ratio=np.array([np.median(v[f'scale_{m}_rank'])/np.median(v[f'scale_{m}_group']) for m in counts])
            ci=np.array([np.percentile(np.median(v[f'scale_{m}_rank'][idx],axis=1)/np.median(v[f'scale_{m}_group'][idx],axis=1),[2.5,97.5]) for m in counts])
            axes[1].errorbar(counts,ratio,yerr=np.abs(np.stack([ratio-ci[:,0],ci[:,1]-ratio])),marker='o',color=colors[i],label=label,capsize=3)
        axes[0].set_xticks([0,1],['4:2 groups','3:3 groups']);axes[0].set(title='Six 9-MiB fetches',ylabel='Makespan median (ms)',ylim=(0,None));axes[0].legend()
        axes[1].axhline(1,color='#555555',linestyle='--');axes[1].set(title='Rank/group ratio by miss count',xlabel='M',ylabel='Ratio of makespan medians');axes[1].legend()
        fig.suptitle('144-MiB versus full 54-GiB NUMA-shared pinned sources')
        pdf.savefig(fig);fig.savefig(a.out/'preview.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(status='PASS',six_fetch=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--small',type=Path,required=True);p.add_argument('--compact',type=Path,required=True);p.add_argument('--spread',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
