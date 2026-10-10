"""Serial ablation, OURS (HAQ) vs RANDOM: R4 input 1024 on both models, and Qwen3 input 512 at R2/R4/R8.
Rebuilds SUMMARY.json and prints the RESULTS tables from raw/.

Every target repeat k>=1 is kept (repeat0 is the in-job warmup). TPS = global requests x 64 / E2E,
the main-table definition.
"""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
CELLS = {  # (part, model, input, gpus) -> raw glob prefix
    ('A', 'Qwen3', 1024, 4): 'qwen_l1024_r4',
    ('A', 'DeepSeekV2Lite', 1024, 4): 'ds_l1024_r4',
    ('B', 'Qwen3', 512, 2): 'qwen_l512_r2',
    ('B', 'Qwen3', 512, 4): 'qwen_l512_r4',
    ('B', 'Qwen3', 512, 8): 'qwen_l512_r8',
}
POLICIES = ('HAQ', 'RANDOM')
NAMES = {'HAQ': 'OURS (HAQ)', 'RANDOM': 'OURS-random'}


def targets(job, policy):
    rows, k = [], 1
    while (job / f'repeat{k}.json').exists():
        s = json.loads((job / f'repeat{k}.json').read_text())
        r0 = json.loads((job / f'repeat{k}_rank0.json').read_text())
        assert (s.get('decode_policy') or r0.get('decode_policy')) == policy, (job, k)
        rows.append(dict(job=job.name, target=k, TTFT=s['TTFT'], TPOT=s['TPOT'], E2E=s['E2E'],
                         TPS=s['global_requests'] * s['output_tokens'] / s['E2E'],
                         global_requests=s['global_requests']))
        k += 1
    assert rows, job
    return rows


def stat(xs):
    return dict(median=st.median(xs), min=min(xs), max=max(xs))


out = dict(status='PASS', units=dict(TPS='tokens/s', TTFT='s', TPOT='s/token', E2E='s'), cells=[])
for (part, model, length, gpus), prefix in CELLS.items():
    for policy in POLICIES:
        jobs = sorted(RAW.glob(f'{prefix}_{policy}_r?'))
        rows = [r for j in jobs for r in targets(j, policy)]
        out['cells'].append(dict(part=part, model=model, input_tokens=length, gpus=gpus, policy=policy,
                                 n=len(rows), jobs=[j.name for j in jobs], samples=rows,
                                 **{k: stat([r[k] for r in rows]) for k in ('TPS', 'TTFT', 'TPOT', 'E2E')}))
(HERE / 'SUMMARY.json').write_text(json.dumps(out, indent=2) + '\n')

fmt = lambda c, k, d: f"{c[k]['median']:.{d}f} [{c[k]['min']:.{d}f}, {c[k]['max']:.{d}f}]"
for part, title in (('A', 'Part A: R4, ShareGPT, C30, B16, input 1024'), ('B', 'Part B: Qwen3, ShareGPT, C30, B16, input 512')):
    print(f'\n### {title} (median [min, max], n)\n')
    print('| Model | GPUs | Method | TPS ↑ | TTFT (s) ↓ | TPOT (s/token) ↓ | n | TPOT vs OURS |')
    print('|---|---:|---|---|---|---|---:|---:|')
    cells = [c for c in out['cells'] if c['part'] == part]
    for c in cells:
        base = next(b for b in cells if b['model'] == c['model'] and b['gpus'] == c['gpus'] and b['policy'] == 'HAQ')
        delta = '—' if c is base else f"{100 * (c['TPOT']['median'] / base['TPOT']['median'] - 1):+.1f}%"
        print(f"| {c['model']} | {c['gpus']} | {NAMES[c['policy']]} | {fmt(c, 'TPS', 2)} | {fmt(c, 'TTFT', 3)} | {fmt(c, 'TPOT', 4)} | {c['n']} | {delta} |")
