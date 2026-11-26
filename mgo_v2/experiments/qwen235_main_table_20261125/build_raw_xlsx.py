"""Raw-data workbook for the Qwen3-235B main-table cell (R4 C30 ShareGPT B16 L512 O64).

Sheets: README, Summary (= summarize.py medians), Trials (every repeat incl. the untimed warmup),
Per_rank (OURS / DeepSpeed rank receipts), Token_timeline (per-token ready time since release),
Memory_raw (per GPU, per repeat, each system's own records), Memory_summary, GPU_smi_max.
Run summarize.py first (it writes SUMMARY.json).
"""
import json, re, sys
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import R, SYSTEMS, passed, repeats, tps, G  # noqa: E402  (also rewrites SUMMARY.json)

HERE = Path(__file__).resolve().parent
OUT = HERE / 'qwen235_main_table_raw_20261125.xlsx'
S = json.loads((HERE / 'SUMMARY.json').read_text())
MODEL = lambda name: 'Qwen3-30B-A3B-Instruct-2507' if '30B' in name else 'Qwen3-235B-A22B-Instruct-2507'
GPUS = [0, 1, 4, 5]
SEL = [(n, [j for j in js if passed(j)]) for n, js in SYSTEMS]

wb = Workbook()


def sheet(title, header, rows, widths=None):
    ws = wb.create_sheet(title)
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor='305496')
        c.alignment = Alignment(wrap_text=True, vertical='top')
    for r in rows:
        ws.append(r)
    ws.freeze_panes = 'A2'
    for i, h in enumerate(header, 1):
        ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(h, max(12, min(40, len(str(h)) + 2)))
    return ws


# README
ws = wb.active
ws.title = 'README'
for line in [
    'Qwen3-235B-A22B-Instruct-2507 main-table cell: R4 (physical GPUs 0/1/4/5, H100 80GB HBM3), C30 expert-cache budget,',
    'ShareGPT, input 512, output 64, per-rank batch 16 (64 global requests), greedy, BF16. Run 2026-11-25..26.',
    'Each job: 1 untimed warmup batch (repeat 0, disjoint requests) + timed target batches (repeat >= 1), cold cache per repeat.',
    'TPS = 64 requests x 64 tokens / E2E (main-table definition). TTFT/TPOT/E2E in seconds; TPOT = (last - first token) / 63.',
    'DeepSpeed ran 5 timed trials (TPOT spread > 5% after 3); all other systems 3.',
    'OURS variants use one shared pinned host copy of the experts in /dev/shm (approved deviation; 4 private copies do not fit host RAM).',
    'Qwen3-30B main_OURS row = reference for the memory footprint on the 30B main-table cell, current code (TPOT faster than the',
    '  published main table, 0.469 vs 0.637 s, because of optimizations that landed after it).',
    'Memory: per GPU at the HBM peak; each system is measured from its own records (see Memory_raw "source" column).',
    '  MoE-Infinity dense is an estimate (native allocator outside torch): nvidia-smi max - EAM peak-charged experts - torch reserved.',
    'Raw receipts: /home/hwlee/mgo-results/qwen235_main_table_20261125/jobs/<job>/repeat<k>[_rank<r>].json',
    'Code: heonniy/mgo branch claude/qwen235-main-table-20261125, mgo_v2/experiments/qwen235_main_table_20261125/',
]:
    ws.append([line])
ws.column_dimensions['A'].width = 140

# Summary
rows = []
for t in S['timing']:
    rows.append([MODEL(t['system']), t['system'], t['n'], ', '.join(t['jobs'])]
                + [t[k][s] for k in ('throughput', 'TTFT', 'TPOT', 'E2E') for s in ('median', 'min', 'max')])
sheet('Summary', ['model', 'system', 'timed trials', 'jobs']
      + [f'{k} {s}' for k in ('TPS', 'TTFT (s)', 'TPOT (s/token)', 'E2E (s)') for s in ('median', 'min', 'max')],
      rows, {'system': 32, 'model': 30, 'jobs': 50})

# Trials (incl. warmup)
rows = []
for name, jobs in SEL:
    for j in jobs:
        r0 = R / j / 'repeat0.json'
        reps = ([(0, json.loads(r0.read_text()))] if r0.exists() else []) + repeats(j)
        for k, r in reps:
            rows.append([MODEL(name), name, j, k, 'warmup (untimed)' if k == 0 else 'timed', r['TTFT'], r['TPOT'], r['E2E'],
                         tps(r), r.get('global_requests', 64), r.get('output_tokens', 64), r.get('status', '')])
sheet('Trials', ['model', 'system', 'job', 'repeat', 'kind', 'TTFT (s)', 'TPOT (s/token)', 'E2E (s)', 'TPS', 'global requests',
                 'output tokens', 'status'], rows, {'system': 32, 'model': 30, 'job': 34})

# Per-rank receipts (OURS, DeepSpeed)
rows = []
for name, jobs in SEL:
    for j in jobs:
        for k, _ in repeats(j):
            for rk in range(4):
                p = R / j / f'repeat{k}_rank{rk}.json'
                if not p.exists():
                    continue
                x = json.loads(p.read_text())
                ttft = (x['first_ns'] - x['release_ns']) / 1e9
                tpot = (x['end_ns'] - x['first_ns']) / 1e9 / 63
                rows.append([MODEL(name), name, j, k, rk, x.get('physical_gpu', GPUS[rk]), ttft, tpot, (x['end_ns'] - x['release_ns']) / 1e9,
                             x['peak_allocated_bytes'] / G, x['peak_reserved_bytes'] / G, x.get('host_rss_bytes', 0) / G,
                             x.get('pinned_host_bytes', 0) / G, x.get('argmax_hash', ''), x.get('finite_logits')])
sheet('Per_rank', ['model', 'system', 'job', 'repeat', 'rank', 'physical GPU', 'rank TTFT (s)', 'rank TPOT (s/token)', 'rank E2E (s)',
                   'peak allocated (GiB)', 'peak reserved (GiB)', 'host RSS (GiB)', 'pinned host (GiB)', 'argmax hash', 'finite logits'],
      rows, {'system': 32, 'model': 30, 'job': 34, 'argmax hash': 66})

# Token timeline
rows = []
for name, jobs in SEL:
    for j in jobs:
        for k, r in repeats(j):
            rank_files = [R / j / f'repeat{k}_rank{rk}.json' for rk in range(4)]
            if all(p.exists() for p in rank_files):
                srcs = [(rk, json.loads(p.read_text())) for rk, p in enumerate(rank_files)]
            else:
                srcs = [('all', r)]
            for rk, x in srcs:
                rel, ready = x['release_ns'], x['token_ready_ns']
                for i, t in enumerate(ready):
                    rows.append([MODEL(name), name, j, k, rk, i, (t - rel) / 1e9, (t - ready[i - 1]) / 1e9 if i else None])
sheet('Token_timeline', ['model', 'system', 'job', 'repeat', 'rank', 'token index', 'ready since release (s)', 'step time (s)'],
      rows, {'system': 32, 'model': 30, 'job': 34})

# Memory raw
rows = []
H = ['model', 'system', 'job', 'repeat', 'rank / GPU', 'source', 'dense (GiB)', 'GPU-resident routed experts (GiB)', 'expert cache capacity (GiB)',
     'resident experts (count)', 'KV cache (GiB)', 'workspace / buffers (GiB)', 'peak allocated (GiB)', 'peak reserved (GiB)', 'note']
for name, jobs in SEL:
    for j in jobs:
        if 'OURS' in name:
            for k, _ in repeats(j):
                for rk in range(4):
                    f = json.loads((R / j / f'repeat{k}_rank{rk}.json').read_text())['memory_footprint']
                    rows.append([MODEL(name), name, j, k, rk, 'memory_footprint receipt', f['dense_weight_bytes'] / G,
                                 f['resident_expert_bytes_decode'] / G, f['expert_cache_capacity_bytes'] / G, f['resident_expert_count_decode'],
                                 f['kv_cache_bytes'] / G, f['workspace_bytes_derived'] / G, f['peak_allocated_bytes'] / G,
                                 f['peak_reserved_bytes'] / G, f['workspace_definition']])
        elif name.startswith('DeepSpeed'):
            ep = max(json.loads((R / jj / f'calibration_rank{rk}.json').read_text())['expert_peak_bytes'] for jj in jobs for rk in range(4))
            for k, _ in repeats(j):
                for rk in range(4):
                    x = json.loads((R / j / f'repeat{k}_rank{rk}.json').read_text())
                    cal = json.loads((R / j / f'calibration_rank{rk}.json').read_text())['expert_peak_bytes']
                    rows.append([MODEL(name), name, j, k, rk, 'ZeRO-3 live-parameter counter', (x['all_parameter_peak_bytes'] - ep) / G,
                                 cal / G, x['parameter_budget_bytes'] / G, None, x['kv_cache_bytes'] / G,
                                 (x['peak_allocated_bytes'] - x['all_parameter_peak_bytes'] - x['kv_cache_bytes']) / G,
                                 x['peak_allocated_bytes'] / G, x['peak_reserved_bytes'] / G,
                                 'experts = gathered per MoE block (transient peak, calibration); dense = all-param live peak - expert peak'])
        elif name.startswith('llama'):
            b = json.loads((R / j / 'buffer_sizes.json').read_text())
            cfg = json.loads((R / j / 'config.json').read_text())
            num = lambda lines: [float(re.search(r'= +([0-9.]+) MiB', s).group(1)) / 1024 for s in lines if 'CUDA' in s and 'Host' not in s and '=' in s]
            model, kvb, comp = num(b['model_buffers']), num(b['kv_buffers']), num(b['compute_buffers'])
            for g in range(4):
                e = cfg['static_expert_bytes_per_gpu'][g] / G
                rows.append([MODEL(name), name, j, 'all', g, 'llama.cpp buffer sizes', model[g] - e, e, None, 7 * 128, kvb[g], comp[g],
                             None, None, 'model buffer = dense layer share + 7 whole expert layers; no torch'])
        else:
            reps = repeats(j)
            for k, r in reps:
                c = r['cache_after']
                for g in range(4):
                    rows.append([MODEL(name), name, j, k, g, 'MoE-Infinity EAM + torch', None, c[f'gpu_{g}_peak_charged_bytes'] / G,
                                 r['expert_budget_per_gpu'][g] / G, None, r['kv_cache_bytes_per_gpu'][g] / G, None,
                                 r['peak_allocated_bytes'][g] / G, r['peak_reserved_bytes'][g] / G,
                                 'experts/dense in native allocator; torch peaks = KV + activations only; dense estimate in Memory_summary'])
sheet('Memory_raw', H, rows, {'system': 32, 'model': 30, 'job': 34, 'source': 30, 'note': 80})

# Memory summary
sheet('Memory_summary', ['model', 'system', 'dense (GiB)', 'GPU-resident routed experts (GiB)', 'KV cache (GiB)', 'runtime buffers / workspace (GiB)',
                         'peak allocated (GiB)', 'peak reserved (GiB)', 'nvidia-smi max (GiB)', 'expert-cache residency'],
      [[MODEL(m['system']), m['system'], m['dense'], m['experts'], m['kv'], m['workspace'], m['alloc'], m['reserved'], float(m['smi']), m['residency']]
       for m in S['memory']], {'system': 32, 'model': 30})

# nvidia-smi max per GPU
rows = []
for name, jobs in SEL:
    for j in jobs:
        mx = {}
        for line in open(R / j / 'resources.jsonl'):
            for gpu in json.loads(line)['gpus']:
                mx[gpu['gpu']] = max(mx.get(gpu['gpu'], 0), gpu['used_mib'])
        rows.append([MODEL(name), name, j] + [mx.get(g, 0) / 1024 for g in GPUS])
sheet('GPU_smi_max', ['model', 'system', 'job'] + [f'GPU{g} max used (GiB)' for g in GPUS], rows, {'system': 32, 'model': 30, 'job': 34})

for ws in wb.worksheets[1:]:
    for row in ws.iter_rows(min_row=2):
        for c in row:
            if isinstance(c.value, float):
                c.number_format = '0.0000'
wb.save(OUT)
print(OUT, {ws.title: ws.max_row - 1 for ws in wb.worksheets})
