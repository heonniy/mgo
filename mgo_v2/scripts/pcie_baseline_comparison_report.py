"""Build a measured one-cell table using G-NEAR and native external baselines."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from pcie_host import write
from pcie_receipts import read_receipt

PKG=Path(__file__).resolve().parents[1];REPO=PKG.parent
EXP=PKG/'experiments/pcie_topology_ablation_20261009'


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    ours=read_receipt(EXP/'stage2_grouped/cohort/live_cohort_validation.json')
    assert ours['status']=='PASS' and len(ours['arms'])==7
    rows=[]
    order=('G-NEAR','R-NEAR','G-BR','G-CA','G-NUMA-CA','R-BR','R-CA')
    for name in order:
        r=ours['arms'][name]['live'];assert r['status']=='PASS'
        rows.append(dict(system=name,status='PASS',repeats=r['primary_repeats'],statistics=r['primary'],
            execution='grouped decode/native C++; globally serialized forward/H2D/compute/return',
            source=f'stage2_grouped/{name}',measured_before_new_baselines=True))
    for name in ('deepspeed','infinity','llama'):
        root=EXP/'baselines'/name
        try:r=read_receipt(root/'baseline_validation.json')
        except FileNotFoundError:
            rows.append(dict(system=name,status='PENDING'));continue
        assert r['status']=='PASS' and r['system']==name
        assert r['workload_sha256']['target']=='dea64853847b1ba1d4f3e0c8cc112eb8ba84dcde5f1d04fb90adb6fca91d67d6'
        rows.append(dict(system=name,status='PASS',repeats=5,statistics=r['statistics'],resource=r['resource'],
            execution='native baseline schedule',source=f'baselines/{name}',pinned_scope=r['pinned_scope']))
    complete=all(r['status']=='PASS' for r in rows)
    result=dict(status='PASS' if complete else 'INCOMPLETE',cell='R4_C30_B16_L512_O64',
        physical_gpus=[0,1,4,5],local_batch=16,global_requests=64,input_tokens=512,output_tokens=64,
        proposed='G-NEAR',rows=rows,
        scope='Original frozen ShareGPT headline64; selected best-input search is a separate workload. Existing validated OURS primaries precede these new baselines; timing-session drift is not controlled.',
        schedule_note='GR means G-NEAR with grouped GEMM and C++ metadata/controller. Baselines preserve native scheduling; no identical-overlap or quota-only causal claim.')
    write(a.out/'RESULTS.json',result)
    with (a.out/'main_table.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['system','status','repeats','TTFT_median_s','TPOT_median_s','E2E_median_s','TPOT_sd_s','execution'])
        for r in rows:
            s=r.get('statistics');w.writerow([r['system'],r['status'],r.get('repeats'),
                *([s[k]['median'] for k in ('TTFT','TPOT','E2E')]+[s['TPOT']['sd']] if s else [None]*4),r.get('execution')])
    lines=['# R4 B16 input512 ShareGPT: G-NEAR and native baselines','',result['scope'],'',result['schedule_note'],'',
        '| System | State | Repeats | TTFT (s) | TPOT (s/token) | E2E (s) | TPOT SD |',
        '|---|---|---:|---:|---:|---:|---:|']
    tex=[r'\begin{tabular}{lrrrr}',r'\hline',r'System & TTFT (s) & TPOT (s) & E2E (s) & Repeats \\',r'\hline']
    for r in rows:
        s=r.get('statistics')
        if s:
            values=[s[k]['median'] for k in ('TTFT','TPOT','E2E')]
            lines.append(f"| {r['system']} | PASS | {r['repeats']} | "+' | '.join(f'{v:.6f}' for v in values+[s['TPOT']['sd']])+' |')
            tex.append(r['system'].replace('_',r'\_')+' & '+' & '.join(f'{v:.6f}' for v in values)+f" & {r['repeats']} "+r'\\')
        else:lines.append(f"| {r['system']} | PENDING | — | — | — | — | — |")
    tex.extend([r'\hline',r'\end{tabular}'])
    (a.out/'main_table.tex').write_text('\n'.join(tex)+'\n')
    lines+=['','All64 requests generate64 tokens, ignoring EOS stopping. Local batches use four GPUs0,1,4,5. OURS has1843 physical expert slots and108GiB unique NUMA-shared pinned source; baseline residency and pinning follow their measured native implementations.',
        '', 'DeepSpeed is native ZeRO-Inference CPU parameter offload with48 MoE leaves, not FastGen. MoE-Infinity is the recorded repaired Qwen3 adaptation. llama.cpp is BF16 synchronous balanced3/32 threads with CUDA graphs/reuse OFF and307 unutilized C30 static expert slots.',
        '', 'Every baseline retains five unfiltered repetitions, workload hashes, numeric/KV/cache-budget checks, measured HBM/RSS and pinning scope in its own archive. Unknown llama staging-pinned memory stays unknown. Same-input generated-token agreement is reproducibility evidence, not task accuracy.']
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if a.commit:
        assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
        files=[a.out/name for name in ('RESULTS.json','RESULTS.md','main_table.csv','main_table.tex')]
        subprocess.run(['git','add','--',*[str(p.relative_to(REPO)) for p in files]],cwd=REPO,check=True)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Update measured R4 B16 input512 ShareGPT table with G-NEAR and native baselines'],cwd=REPO,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',action='store_true');main(p.parse_args())
