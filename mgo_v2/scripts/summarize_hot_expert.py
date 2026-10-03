#!/usr/bin/env python3
"""Final evidence and limits for the bounded hot-expert threshold study."""
from hot_expert_cpu import P,ROOT,sha,write
import json,sys,time
from pathlib import Path
import numpy as np

def main():
    h0=json.loads((P/'H0.json').read_text());h1=json.loads((P/'H1.json').read_text());h2=json.loads((P/'H2_H4.json').read_text())
    assert h0['status']==h1['status']==h2['status']=='PASS'
    manifest=json.loads((P/'execution_manifest.json').read_text());correction=json.loads((P/'startup_correction_manifest.json').read_text())
    expected=dict(manifest['source_sha256']);expected.update(correction['source_sha256'])
    for f,h in expected.items():assert sha(f)==h,f
    prev=P.parent/'future_rank_affinity_placement_20261003/validation.json'
    assert sha(prev)==manifest['previous_validation_sha256']
    for r in h0['receipts']+h1['receipts']:assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    assert not h2['H3_cells'],'Use the bounded H3 runner if measured concurrent thresholds have candidates'
    write(P/'H3.json',dict(status='SKIPPED_NO_CANDIDATES',cells=0,condition='four-rank-concurrent',reason='T0 median/p90 >512; R3 median/p90 512; all existing remote n <=32. Single-rank p90 sensitivity remains reported in H2.',future_prediction=False))
    distribution=json.loads((P/'hotness_distribution.json').read_text())
    savings=json.loads((P/'hotness_peer_saving.json').read_text())
    micro=json.loads((P/'break_even_microcost.json').read_text());coverage=json.loads((P/'threshold_coverage.json').read_text())
    assert len(micro['rows'])==62 and len(micro['thresholds'])==8 and len(coverage['rows'])==24
    assert all(r['samples']==(60 if r['kind']=='pair-pooled' else 30) for r in micro['rows'])
    threshold_lookup={(r['mode'],r['h2d_condition'],r['metric']):r for r in micro['thresholds']}
    for r in coverage['rows']:
        t=threshold_lookup[r['mode'],r['h2d_condition'],r['metric']]
        assert r['n_star']==t['n_star_display']
        assert 0<=r['candidate_fraction']<=1 and 0<=r['remote_route_fraction']<=1 and 0<=r['marginal_peer_fraction']<=1
        assert sum(r['layer_distribution'].values())==r['crossing_occurrences']
        if t['n_star'] is None or t['n_star']>r['batch']:assert r['crossing_occurrences']==0
    assert h2['decision']=='CURRENT_BATCH_TOO_COLD' and 2*h2['B32_max_remote']<threshold_lookup['R3','single-rank','median']['n_star']
    modes=[]
    for cell in h1['cells']:
        state=json.loads((ROOT/'H1'/cell['label']/'status.json').read_text());assert state['status']=='PASS'
        modes.append(state)
    host_min=min(s['host_available_bytes'] for state in modes for s in state['memory'])
    group_peak=max(s.get('group_rss_bytes',0) for state in modes for s in state['memory'])
    workers=[]
    load=Path('/home/hwlee/mgo-results/model_inference_load_20261003')
    for w in h1['restored_workers']:
        assert w['status']=='STARTED' and w['gpu'] in (0,1,4,5)
        assert 'model_inference_load.py' in Path(f"/proc/{w['pid']}/cmdline").read_bytes().decode()
        r=json.loads((load/f"gpu{w['gpu']}.json").read_text())
        assert r['pid']==w['pid'] and r['batch']==1024 and time.time()-r['unix']<120
        workers.append({k:r[k] for k in ('gpu','pid','batch','iterations','unix')})
    assert {w['gpu'] for w in workers}=={0,1,4,5}
    for w in h1['workers_before']:
        if w['gpu'] in (2,3,6,7):assert not Path(f"/proc/{w['pid']}").exists()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,cdf)=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for mode,color in [('T0','#2563eb'),('R3','#d97706')]:
        for pass_id,style in [(0,'-'),(1,'--')]:
            rows=sorted([r for r in micro['rows'] if r['kind']=='pair' and r['mode']==mode and r['pass_id']==pass_id],key=lambda r:r['case'])
            ax.plot([r['case'] for r in rows],[r['median_ms'] for r in rows],style,marker='.',color=color,label=f'{mode} pass {pass_id+1}')
    hm=threshold_lookup['R3','four-rank-concurrent','median']['h2d_ms'];hp=threshold_lookup['R3','four-rank-concurrent','p90']['h2d_ms']
    ax.axhline(hm,color='#555555',label='4-rank H2D median');ax.axhline(hp,color='#555555',linestyle=':',label='4-rank H2D p90')
    ax.set_xscale('log',base=2);ax.set(xlabel='Rows per remote dispatch + combine pair',ylabel='Max-rank CUDA interval (ms)',title='Measured medians: substantial pass dependence');ax.legend(fontsize=8);ax.grid(alpha=.2)
    for batch in (8,16,32):
        rows=sorted([r for r in savings['rows'] if r['batch']==batch],key=lambda r:r['n']);weights=np.array([r['occurrences'] for r in rows]);cdf.step([r['n'] for r in rows],np.cumsum(weights)/weights.sum(),where='post',label=f'B{batch}')
    cdf.set(xlabel='Current rank-local remote demand n',ylabel='Remote occurrence cumulative fraction',title='Existing traces: maximum demand = 32',xlim=(1,33),ylim=(0,1.02));cdf.legend();cdf.grid(alpha=.2)
    fig.savefig(P/'microcost_hotness.png',dpi=180);fig.savefig(P/'microcost_hotness.svg');plt.close(fig)
    svg=P/'microcost_hotness.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    lines=['# Hot-expert replication break-even','',
        '**CURRENT_BATCH_TOO_COLD under the frozen pooled-median H4 rule.** This is not a robust universal threshold: R3 has substantial pass/order dependence. No H3 cell or larger-batch capture was run.','',
        'The preceding single-copy placement packet completed at `85283b0` before H0/H1 started. This packet uses existing traces, then model-free pair communication and one-expert pinned H2D calibration on GPU 0/1/4/5. R3 is synthetic P2P-disabled SHM, not a physical PCIe-only server.','',
        '## Current rank-local hotness','', '| Local batch | Remote n p50 | p90 | p95 | p99 | Max |','|---:|---:|---:|---:|---:|---:|']
    for r in distribution['rows']:
        if r['subset']=='remote':lines.append(f"| {r['batch']} | {r['p50']:g} | {r['p90']:g} | {r['p95']:g} | {r['p99']:g} | {r['max']} |")
    lines+=['','Every candidate satisfies exact current saving ≤ 2×n×4096 bytes. Thus even n=32 can save at most 256 KiB of current peer traffic, versus a 9-MiB expert fetch. This is byte accounting only; equal bytes do not imply equal costs. Individual marginal dispatch savings interact, so coverage uses their explicitly labeled nonadditive sum.','',
        '## Measured crossover rows','', '| Mode | H2D condition | Metric | H2D ms | Pooled n* | Pass 1 n* | Pass 2 n* |','|:---|:---|:---|---:|:---|:---|:---|']
    for t in micro['thresholds']:
        p=t['per_pass_n_star'];lines.append(f"| {t['mode']} | {t['h2d_condition']} | {t['metric']} | {t['h2d_ms']:.6f} | {t['n_star_display']} | {p['0'] if p['0'] is not None else '>512'} | {p['1'] if p['1'] is not None else '>512'} |")
    lines+=['','Pair passes were T0→R3 then R3→T0, with ten warmups and thirty measured samples per n/pass; pooled estimates use sixty paired max-rank samples. H2D has ten warmups and thirty samples per condition. No linear extrapolation, extra sizes, retuning or statistical repetitions were added.','',
        '**Important sensitivity:** R3 single-pass crossover is 512 in the first pass and 1 in the second. Pooled single-rank p90 also crosses at n=1, covering all remote occurrences. The preselected four-rank concurrent H2D p90 is higher, so its pooled crossover is 512. These differences prevent treating 512 as a stable intrinsic hardware threshold or claiming an established monotonic bandwidth crossover. The measured CUDA interval includes communication launch pacing and scheduling; it is not pure wire-transfer time.','',
        '![Microcost and observed hotness](microcost_hotness.png)','',
        '## Coverage, replay gate and larger-batch decision','',
        '- H2 reports all eight transport/H2D/metric combinations for every batch, including the single-rank p90 n=1 result. That result has 100% remote candidate/route/marginal-byte coverage; all other pooled combinations have zero coverage in current traces.',
        '- Before measurement, H3 was assigned the four-rank concurrent H2D condition, yielding at most four thresholds per batch. Both T0 thresholds are >512, both R3 thresholds are 512, and no existing remote demand exceeds 32. H3 is therefore SKIPPED_NO_CANDIDATES; no threshold-policy or frontier result is fabricated.',
        '- H4 uses the optimistic single-rank R3 pooled median n*=512. B32 maximum 32 is below 0.5×512=256, so the predeclared result is CURRENT_BATCH_TOO_COLD. B8→B16→B32 p99/max grows monotonically, but the magnitude gate fails. No B64 proposal/capture or B128 capture is initiated.',
        '- The isolated thresholds exclude victim eviction and reload cost, so they are optimistic for replication. The current evidence does not establish an actionable hot/cold controller, especially given the R3 pass sensitivity. Stop without extra timing or GPU/model experiments.','',
        '## Startup correction and validation','',
        '- Both transport smokes passed: P2P/IPC for T0 and SHM/direct/direct for R3. INFO logging was confined to those smokes; timing processes had no NCCL debug environment.',
        '- The first timing process failed an environment assertion before any timing/NCCL initialization because bootstrap injected NCCL_P2P_DISABLE=0. Failure evidence was committed at ed8e912; 597b224 applied the same post-bootstrap cleanup as the existing validated workers. Successful smokes were retained, and only the five unexecuted measurement processes ran. Failed data was not overwritten.',
        '- All four completed pair passes and both H2D conditions passed payload checks outside timing. The 40 pair-size/pass cells contain 1,200 timed pairs; H2D contains 60 timed samples. All raw evidence has size/SHA256 receipts.',
        f"- H0 CPU peak RSS {h0['peak_rss_mib']:.2f} MiB, H2 peak RSS {h2['peak_rss_mib']:.2f} MiB, each under 4-GiB address space. H1 peak sampled process-tree RSS {group_peak/2**30:.2f} GiB; minimum host available {host_min/2**30:.2f} GiB. No OOM occurred.",
        '- One exact marginal accounting unit test and two crossover boundary/nonmonotonicity tests passed. Existing F reproduction from the completed placement packet remains hash-identical.',
        '- Model workers were restored on GPU 0/1/4/5 at local batch1024; live receipts were checked. Owner-stopped workers on GPU 2/3/6/7 were not restarted. The post-experiment snapshot shows other processes on those GPUs, which were not touched; that snapshot does not establish their timing relative to calibration.','',
        'See [hotness](hotness_distribution.json), [marginal savings](hotness_peer_saving.json), [microcosts](break_even_microcost.json), [coverage](threshold_coverage.json), [H4 gate](H2_H4.json), [validation](validation.json), and [execution conventions](EXECUTION_PROTOCOL.md).','']
    (P/'RESULTS.md').write_text('\n'.join(lines))
    assert 'torch' not in sys.modules
    validation=dict(status='PASS',decision=h2['decision'],robust_transport_threshold_established=False,pass_dependence=True,H0_batches=[8,16,32],transport_smokes=2,pair_cells=40,timed_pairs=1200,h2d_conditions=2,h2d_samples=60,startup_failures_before_timing=1,H3_cells=0,new_model_captures=0,B64_capture=False,B128_capture=False,unit_tests=3,H0_peak_rss_mib=h0['peak_rss_mib'],H2_peak_rss_mib=h2['peak_rss_mib'],H1_peak_sampled_group_rss_bytes=group_peak,H1_min_host_available_bytes=host_min,source_hashes_verified_with_documented_startup_correction=True,workers_restored=workers,owner_stopped_workers_not_restarted=True,analysis_source_sha256={str(f):sha(f) for f in (Path(__file__).resolve(),Path(__file__).with_name('hot_expert_thresholds.py').resolve())})
    validation['output_sha256']={f.name:sha(f) for f in P.iterdir() if f.suffix in ('.json','.csv','.svg','.png') and f.name!='validation.json'}|{'RESULTS.md':sha(P/'RESULTS.md')}
    write(P/'validation.json',validation)
    print(json.dumps({k:validation[k] for k in ('status','decision','H3_cells','robust_transport_threshold_established','workers_restored')},indent=2))
if __name__=='__main__':main()
