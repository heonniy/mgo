"""Cache-size sweep ablation (C20/C30/C40): OURS (HAQ) vs RANDOM in serial ablation mode, plus the
existing baseline sweeps on the same frozen manifests. Rebuilds SUMMARY.json and prints the RESULTS tables.

Serial-mode samples come from raw/ (every target repeat k>=1; nothing dropped). Baseline rows are read
unchanged from ../qwen_cache_ablation_20261009 and ../deepseek_cache_ablation_20261009.
"""
import csv, json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
EXP = HERE.parent
CACHES = (20, 30, 40)
SERIAL = {
    'Qwen3': {**{('HAQ', c): sorted(RAW.glob(f'qwen_c{c}_HAQ_r?')) for c in CACHES},
              **{('RANDOM', c): sorted(RAW.glob(f'qwen_c{c}_RANDOM_r?')) for c in CACHES}},
    'DeepSeekV2Lite': {('HAQ', 20): sorted(RAW.glob('ds_c20_HAQ_r?')), ('HAQ', 40): sorted(RAW.glob('ds_c40_HAQ_r?')),
                       ('HAQ', 30): [RAW / 'ds_haq_serial_c30_v1', RAW / 'ds_static_c30_v1'],
                       **{('RANDOM', c): sorted(RAW.glob(f'ds_c{c}_RANDOM_r?')) for c in CACHES}},
}
NAMES = {'HAQ': 'OURS (HAQ, serial)', 'RANDOM': 'RANDOM (serial)'}


def targets(job, policy):
    rows, k = [], 1
    while (job / f'repeat{k}.json').exists():
        s = json.loads((job / f'repeat{k}.json').read_text())
        ranks = [json.loads((job / f'repeat{k}_rank{r}.json').read_text()) for r in range(4)]
        if (s.get('decode_policy') or ranks[0].get('decode_policy')) == policy:
            if 'native_executor_counts' in ranks[0]:
                copies = [x['native_executor_counts']['serial_waited_copies'] / 63 for x in ranks]
            else:
                copies = [x / 63 for x in ranks[0]['decode_fetches_per_rank']]
            rows.append(dict(job=job.name, target=k, TPOT=s['TPOT'], TTFT=s['TTFT'], E2E=s['E2E'],
                             argmax_hash=ranks[0]['argmax_hash'], copies_per_token_by_rank=copies))
        k += 1
    return rows


def stat(xs):
    return dict(median=st.median(xs), min=min(xs), max=max(xs))


def baselines(model):
    """{(system, cache): {TTFT/TPOT/E2E: {median,min,max}, n, source}} from the committed sweeps."""
    out = {}
    if model == 'Qwen3':
        d = EXP / 'qwen_cache_ablation_20261009'
        names = {'deepspeed': 'DeepSpeed ZeRO-3 offload', 'infinity': 'MoE-Infinity (repaired)', 'llama': 'llama.cpp balanced'}
        for r in json.loads((d / 'BASELINE_SWEEP_RESULTS.json').read_text())['rows']:
            if r['cache_percent'] in CACHES:
                m = r['metrics']
                out[(names[r['system']], r['cache_percent'])] = dict(
                    n=len(m['TPOT']['samples']), source=r['label'],
                    **{k: dict(median=m[k]['median'], min=m[k]['minimum'], max=m[k]['maximum']) for k in ('TTFT', 'TPOT', 'E2E')})
        for r in csv.DictReader(open(d / 'RESULTS.csv')):
            c = int(r['cache_percent'])
            if c in CACHES:
                out[('OURS (default overlap mode)', c)] = row_from_csv(r)
    else:
        d = EXP / 'deepseek_cache_ablation_20261009'
        names = {'DeepSpeed ZeRO-Inference': 'DeepSpeed ZeRO-3 offload', 'MoE-Infinity (repaired)': 'MoE-Infinity (repaired)',
                 'llama.cpp balanced': 'llama.cpp balanced', 'main_OURS': 'OURS (default overlap mode)'}
        for r in csv.DictReader(open(d / 'RESULTS.csv')):
            c = int(r['cache_percent'])
            if c in CACHES:
                out[(names[r['system']], c)] = row_from_csv(r)
    return out


def row_from_csv(r):
    g = lambda k, s: float(r[f'{k}_{s}_s' if k != 'TPOT' else f'TPOT_{s}_s_per_token'])
    return dict(n=None, source=r['selected_attempt'],
                **{k: dict(median=g(k, 'median'), min=g(k, 'min'), max=g(k, 'max')) for k in ('TTFT', 'TPOT', 'E2E')})


out = dict(status='PASS', units=dict(TPOT='s/token', TTFT='s', E2E='s'), models={})
for model, src in SERIAL.items():
    cells = {c: {} for c in CACHES}
    for (policy, c), jobs in src.items():
        rows = [r for j in jobs for r in targets(j, policy)]
        assert rows, (model, policy, c)
        cells[c][NAMES[policy]] = dict(n=len(rows), source=[j.name for j in jobs], targets=rows,
                                       hashes=sorted({r['argmax_hash'] for r in rows}),
                                       **{k: stat([r[k] for r in rows]) for k in ('TTFT', 'TPOT', 'E2E')})
    for (system, c), row in baselines(model).items():
        cells[c][system] = row
    out['models'][model] = cells
(HERE / 'SUMMARY.json').write_text(json.dumps(out, indent=1) + '\n')

ORDER = ['OURS (HAQ, serial)', 'RANDOM (serial)', 'OURS (default overlap mode)',
         'llama.cpp balanced', 'MoE-Infinity (repaired)', 'DeepSpeed ZeRO-3 offload']
for model, cells in out['models'].items():
    print(f'### {model}: serial ablation, OURS vs RANDOM (TPOT s/token, median [min, max], n)\n')
    print('| Cache | OURS (HAQ) | RANDOM | RANDOM vs OURS |')
    print('|---:|---|---|---:|')
    for c in CACHES:
        h, r = cells[c][NAMES['HAQ']], cells[c][NAMES['RANDOM']]
        f = lambda x: f"{x['TPOT']['median']:.4f} [{x['TPOT']['min']:.4f}, {x['TPOT']['max']:.4f}] ({x['n']})"
        print(f"| C{c} | {f(h)} | {f(r)} | {(r['TPOT']['median'] / h['TPOT']['median'] - 1) * 100:+.1f}% |")
    print(f'\n### {model}: all systems (medians)\n')
    print('| System | ' + ' | '.join(f'C{c} TPOT' for c in CACHES) + ' | ' + ' | '.join(f'C{c} TTFT / E2E' for c in CACHES) + ' |')
    print('|---|' + '---:|' * (2 * len(CACHES)))
    for s in ORDER:
        tp = ' | '.join(f"{cells[c][s]['TPOT']['median']:.4f}" for c in CACHES)
        te = ' | '.join(f"{cells[c][s]['TTFT']['median']:.2f} / {cells[c][s]['E2E']['median']:.1f}" for c in CACHES)
        print(f'| {s} | {tp} | {te} |')
    print()
