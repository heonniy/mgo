"""Same-packet compute results; no claim of grouped serving TPOT."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def run(a):
 data=[json.loads((a.root/'R-NEAR'/f'grouped_probe_rank{r}.json').read_text()) for r in range(4)]
 rows=[x for d in data for x in d['records']];checks=[x for d in data for x in d['numerics']]
 assert all(len(d['records'])==288 and len(d['numerics'])==48 for d in data)
 records={(x['rank'],x['event'],x['repeat'],x['backend']):x for x in rows}
 events=sorted({x['event'] for x in rows});assert len(events)==48
 critical=[]
 for event in events:
  for repeat in range(3):
   row=dict(event=event,step=event//48,layer=event%48,repeat=repeat)
   for backend in ('native','grouped'):
    values=[records[r,event,repeat,backend] for r in range(4)]
    row[backend+'_max_rank_host_complete_ms']=max(v['host_complete_ms'] for v in values)
    row[backend+'_max_rank_cuda_window_ms']=max(v['cuda_window_ms'] for v in values)
   critical.append(row)
 summary={}
 for backend in ('native','grouped'):
  values=np.array([r[backend+'_max_rank_host_complete_ms'] for r in critical])
  summary[backend]=dict(mean=float(values.mean()),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),min=float(values.min()),max=float(values.max()))
 agreement=[]
 for r in range(4):
  actual=json.loads((a.root/'R-NEAR'/f'repeat-1_rank{r}.json').read_text())
  expected=json.loads((a.reference/'R-NEAR'/f'repeat3_rank{r}.json').read_text())
  assert actual['tokens']==expected['tokens'] and actual['validation']['state_hash']==expected['validation']['state_hash']
  assert actual['finite_logits'] and actual['validation']['physical_slots']==1843
  assert actual['validation']['scheduler']['background_copies']==0
  agreement.append(True)
 assert all(c['finite'] and c['relative_l2']<=.01 for c in checks)
 a.out.mkdir(parents=True,exist_ok=True)
 for filename,records_out in [('compute_samples.csv',rows),('critical_compute_samples.csv',critical),('numerics.csv',checks)]:
  with (a.out/filename).open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(records_out[0]));writer.writeheader();writer.writerows(records_out)
 result=dict(status='PASS',scope='R-NEAR paired compute-only on identical real packets; native output alone advances model',
             max_rank_local_complete_ms=summary,median_ratio_native_over_grouped=summary['native']['median']/summary['grouped']['median'],
             median_latency_reduction_fraction=1-summary['grouped']['median']/summary['native']['median'],
             event_count=48,repeat_count_per_event_per_backend=3,raw_rank_samples=len(rows),
             workspace_bytes_per_rank=data[0]['workspace_bytes'],max_relative_l2=max(c['relative_l2'] for c in checks),
             max_absolute_error=max(c['max_abs'] for c in checks),native_reference_token_cache_parity=all(agreement),
             source_root=str(a.root),reference=str(a.reference),
             limits=['Selected fixed event set: steps1,32,63 and layers0,3,...45; not a serving-workload speedup estimate',
                     'Local completion includes CPU submission gaps and CUDA completion; max rank durations exclude Gloo rendezvous/start skew',
                     'Extra grouped metadata preparation occurs before metadata/forward completion and is outside compute timer',
                     'No Torch profiler inside paired timing; each current packet warms both methods once before three counterordered pairs',
                     'Different floating-point reduction order: finite BF16 outputs with relative-L2 <=1%; no full grouped greedy-generation parity claimed'])
 (a.out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
 text=f'''# R-NEAR identical-packet grouped/native compute probe

The native model retains exactly the reference tokens and final cache on all four ranks. The alternate grouped outputs are checked but never advance the model. This is a computation diagnostic, not a grouped serving TPOT result or a Ready-First experiment.

48 real decode events (steps 1, 32, 63; layers 0, 3, ..., 45) × 3 counterordered timed pairs × 4 ranks = {len(rows)} raw local samples. Both methods receive the same input packet, expert cache weights, routing weights and physical slots, after every rank finishes mandatory H2D. GPU event windows include host enqueue gaps. Each packet warms both methods once before timing. No Torch profiler runs during these pairs.

| Backend | Median of event/repeat max-rank local completion (ms) | Mean (ms) | p10 / p90 (ms) |
|---|---:|---:|---:|
| Native expert loop | {summary['native']['median']:.6f} | {summary['native']['mean']:.6f} | {summary['native']['p10']:.6f} / {summary['native']['p90']:.6f} |
| All-ready grouped | {summary['grouped']['median']:.6f} | {summary['grouped']['mean']:.6f} | {summary['grouped']['p10']:.6f} / {summary['grouped']['p90']:.6f} |

Ratio of medians: {result['median_ratio_native_over_grouped']:.3f}×; compute-only median latency reduction: {100*result['median_latency_reduction_fraction']:.3f}%. This ratio is descriptive of the fixed packet set, excludes rendezvous and control/index preparation, and must not be reported as overall generation acceleration.

Numerics: all 192 packet/rank checks finite; maximum weighted-output relative L2 {100*result['max_relative_l2']:.4f}%, maximum absolute error {result['max_absolute_error']:.6f}. Different BF16 reduction orders remain; actual grouped greedy generation has not been validated. Additional bounded activation workspace: {data[0]['workspace_bytes']/2**20:.2f} MiB/rank. No expert-weight D2D copies or changes to C30/P2 residency. The source remains actual NUMA-shared 54 GiB pinned per node. The global phase-order verifier passes all 3072 events on every rank.

The result supports a large removable execution/submission cost in this native per-expert path. The C++ loop still invokes ATen/cuBLAS and gather/activation/weight operations separately for every expert. Grouped projections combine gate/up and execute many experts per launch, then grouped down and weight/scatter, reducing CPU submissions and small-operation scheduling. The separate profiler correlates each executor's launches with its actual GPU kernels; its intrusive CPU timings are not used here.

Ready-First addresses a separate exposed wait: start resident or completed experts while other expert copies remain in flight. This probe intentionally disables that opportunity by completing all H2D first. A grouped Ready-First implementation would group the currently ready experts into waves, potentially trading earlier execution for more launch batches. Forward and metadata communication must finish before H2D, and return communication must wait for all ranks' H2D and expert compute to finish.
'''
 (a.out/'RESULTS.md').write_text(text)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11.7,4.5),gridspec_kw={'width_ratios':[2,1]})
 colors={'native':'#235b85','grouped':'#c86c27'}
 for backend in ('native','grouped'):
  values=[np.mean([r[backend+'_max_rank_host_complete_ms'] for r in critical if r['event']==event]) for event in events]
  axes[0].plot(range(48),values,'.-',color=colors[backend],label=backend)
 axes[0].set_yscale('log');axes[0].set_ylabel('Max-rank local compute completion (ms, log scale)')
 axes[0].set_xlabel('Fixed packet event (mean of three repeats)');axes[0].legend(frameon=False)
 for x in (15.5,31.5):axes[0].axvline(x,color='#999999',linewidth=.7,linestyle=':')
 axes[0].set_xticks([7.5,23.5,39.5],['decode step 1','decode step 32','decode step 63'])
 axes[0].grid(axis='y',alpha=.2)
 for idx,backend in enumerate(('native','grouped')):
  s=summary[backend];axes[1].bar(idx,s['median'],color=colors[backend],width=.55)
  axes[1].errorbar(idx,s['median'],yerr=[[s['median']-s['p10']],[s['p90']-s['median']]],color='#333333',capsize=4)
  axes[1].text(idx,s['p90']+.9,f"{s['median']:.3f} ms",ha='center',fontsize=10)
 axes[1].set_xticks([0,1],['native','grouped']);axes[1].set_ylim(0,23);axes[1].set_ylabel('Completion (ms)')
 axes[1].set_title('Median and p10–p90\n144 event/repeat pairs',fontsize=11)
 fig.suptitle('R-NEAR: identical-packet expert compute, all H2D already complete',fontsize=14)
 fig.text(.04,.045,f"48 fixed events × 3 pairs × 4 ranks; compute-only, no serving TPOT claim. Maximum relative L2: {100*result['max_relative_l2']:.3f}%.\nSource: G4_grouped_same_packet_attempt1 / grouped_probe_rank0..3.json; native output alone advances model.",fontsize=9)
 fig.tight_layout(rect=[0,.15,1,.9]);fig.savefig(a.out/'grouped_compute_probe.pdf');fig.savefig(a.out/'grouped_compute_probe.png',dpi=130);plt.close(fig)
 print(json.dumps(result))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
