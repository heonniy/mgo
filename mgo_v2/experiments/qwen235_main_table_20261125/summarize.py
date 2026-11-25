"""Qwen3-235B main-table cell: timing and per-GPU memory footprint for every system.

Timing: every timed target repeat (repeat >= 1) of the selected jobs, including the +2 stability extension
jobs; median and [min, max]. TPS = global requests x 64 / E2E (main-table definition).

Memory, per GPU at the HBM peak of the timed repeats (min-max over GPUs/ranks and repeats). The systems
manage memory differently, so each column is filled from that system's own records:
  OURS variants: memory_footprint receipt (dense, expert cache, KV, derived workspace, torch peaks).
  DeepSpeed: ZeRO-3 gathers parameters per module. "GPU-resident routed experts" is the peak of gathered
    expert parameters (calibration pass, periodic full scans); dense = all-parameter live peak - expert peak;
    workspace = torch peak allocated - all-parameter live peak - KV.
  MoE-Infinity: dense parameters on GPU, KV tensors, EAM resident/peak-charged expert bytes; its expert
    cache is a native allocation outside the torch allocator, so torch peaks exclude it.
  llama.cpp: CUDA model buffer (dense share + whole GPU expert layers), KV buffer, compute buffer; no torch.
Actual GPU usage is the nvidia-smi maximum sampled every second by the job supervisor.
"""
import json, re, statistics as st
from pathlib import Path

R = Path('/home/hwlee/mgo-results/qwen235_main_table_20261125/jobs')
HERE = Path(__file__).resolve().parent
G = 2 ** 30
EB, TOTAL_EXPERTS = 3 * 4096 * 1536 * 2, 94 * 128
SYSTEMS = [  # name, jobs
    ('DeepSpeed ZeRO-Inference', ['q235_deepspeed_r3', 'q235_deepspeed_r3_ext2']),
    ('llama.cpp', ['q235_llama_r3', 'q235_llama_r3_ext2']),
    ('MoE-Infinity (repaired)', ['q235_infinity_r3', 'q235_infinity_r3_ext2']),
    ('main_OURS', ['q235_main_OURS_r3', 'q235_main_OURS_r3_ext2']),
    ('OURS-random (overlap)', ['q235_OURS_random_overlap_r3', 'q235_OURS_random_overlap_r3_ext2']),
    ('OURS-random (serial, def. B)', ['q235_OURS_random_serial_r3', 'q235_OURS_random_serial_r3_ext2']),
    # Reference: Qwen3-30B-A3B main_OURS on the main-table cell (rank-private full pinned source), current code.
    ('Qwen3-30B main_OURS (reference)', ['q30b_main_OURS_memory_r3']),
]


def passed(job):
    s = R / job / 'status.json'
    return s.exists() and json.loads(s.read_text())['status'] == 'PASS'


def repeats(job):
    out, k = [], 1
    while (R / job / f'repeat{k}.json').exists():
        out.append((k, json.loads((R / job / f'repeat{k}.json').read_text())))
        k += 1
    return out


def tps(r):
    # main-table definition: global requests x 64 output tokens / E2E (llama receipts carry no throughput field)
    n = r.get('global_requests', len(r.get('tokens', [])) or 64)
    assert n == 64, n
    return n * 64 / r['E2E']


def smi_max(job):
    peak = 0
    for line in open(R / job / 'resources.jsonl'):
        d = json.loads(line)
        peak = max([peak] + [g['used_mib'] for g in d['gpus']])
    return peak / 1024


def span(v):
    return f'{min(v):.2f}–{max(v):.2f}' if v else 'n/a'


def memory(name, jobs):
    m = {}
    if 'OURS' in name:
        f = [json.loads((R / j / f'repeat{k}_rank{r}.json').read_text())['memory_footprint']
             for j in jobs for k, _ in repeats(j) for r in range(4)]
        res = json.loads((R / jobs[0] / 'result.json').read_text())['memory_by_repeat']['1']
        m.update(dense=span([x['dense_weight_bytes'] / G for x in f]),
                 experts=span([x['resident_expert_bytes_decode'] / G for x in f]) + f" (capacity {span([x['expert_cache_capacity_bytes'] / G for x in f])})",
                 kv=span([x['kv_cache_bytes'] / G for x in f]),
                 workspace=span([x['workspace_bytes_derived'] / G for x in f]),
                 alloc=span([x['peak_allocated_bytes'] / G for x in f]),
                 reserved=span([x['peak_reserved_bytes'] / G for x in f]),
                 residency=f"{res['expert_cache_residency_ratio_decode']:.2%} ({res['resident_expert_count_decode']}/{res['expert_keys']})")
    elif name.startswith('DeepSpeed'):
        rows = [json.loads((R / j / f'repeat{k}_rank{r}.json').read_text()) for j in jobs for k, _ in repeats(j) for r in range(4)]
        cal = [json.loads((R / j / f'calibration_rank{r}.json').read_text()) for j in jobs for r in range(4)]
        ep = [c['expert_peak_bytes'] / G for c in cal]
        live = [x['all_parameter_peak_bytes'] / G for x in rows]
        m.update(dense=span([x['all_parameter_peak_bytes'] / G - max(ep) for x in rows]) + ' (gathered; persistent %.4f)' % (rows[0]['gpu_persistent_param_bytes'] / G),
                 experts=span(ep) + ' (gathered per MoE block, transient)',
                 kv=span([x['kv_cache_bytes'] / G for x in rows]),
                 workspace=span([(x['peak_allocated_bytes'] - x['all_parameter_peak_bytes'] - x['kv_cache_bytes']) / G for x in rows]),
                 alloc=span([x['peak_allocated_bytes'] / G for x in rows]),
                 reserved=span([x['peak_reserved_bytes'] / G for x in rows]),
                 residency=f"{max(ep) * G / (TOTAL_EXPERTS * EB):.2%} of routed-expert bytes at peak (transient)")
    elif name.startswith('llama'):
        b = json.loads((R / jobs[0] / 'buffer_sizes.json').read_text())
        num = lambda lines: [float(re.search(r'= +([0-9.]+) MiB', s).group(1)) / 1024 for s in lines if 'CUDA' in s and 'Host' not in s and '=' in s]
        model, kvb, comp = num(b['model_buffers']), num(b['kv_buffers']), num(b['compute_buffers'])
        cfg = json.loads((R / jobs[0] / 'config.json').read_text())
        per_gpu_experts = cfg['static_expert_bytes_per_gpu'][0] / G
        m.update(dense=span([x - per_gpu_experts for x in model]) + ' (layer-split, not replicated)',
                 experts=f'{per_gpu_experts:.2f} (7 whole layers/GPU; {cfg["unused_expert_budget_bytes"] / G:.2f} GiB budget unused)',
                 kv=span(kvb) + ' (buffer)', workspace=span(comp) + ' (compute buffer)',
                 alloc='n/a (no torch)', reserved='n/a (no torch)',
                 residency=f"{28 * 128 / TOTAL_EXPERTS:.2%} (28 x 128 / {TOTAL_EXPERTS}, static)")
    else:  # MoE-Infinity
        rows = [r for j in jobs for _, r in repeats(j)]
        stats = [r['cache_after'] for r in rows]
        peak_charged = [s[f'gpu_{g}_peak_charged_bytes'] / G for s in stats for g in range(4)]
        # Dense weights live in MoE-Infinity's native allocator (torch sees 0 B of CUDA parameters), so estimate
        # them per GPU as nvidia-smi max - EAM peak-charged experts - torch peak reserved (an upper bound that
        # also contains the CUDA context and any native staging buffers).
        smi_g = [0.0] * 4
        for j in jobs:
            for line in open(R / j / 'resources.jsonl'):
                for i, gpu in enumerate(json.loads(line)['gpus']):
                    smi_g[i] = max(smi_g[i], gpu['used_mib'] / 1024)
        est = [smi_g[g] - max(s[f'gpu_{g}_peak_charged_bytes'] for s in stats) / G - max(r['peak_reserved_bytes'][g] for r in rows) / G for g in range(4)]
        m.update(dense=span(est) + ' (native allocator; est. = smi - experts - torch reserved, incl. CUDA context)',
                 experts=span(peak_charged) + ' (EAM peak charged)',
                 kv=span([d / G for r in rows for d in r['kv_cache_bytes_per_gpu']]),
                 workspace='in torch peak (KV + activations); native buffers in dense estimate',
                 alloc=span([d / G for r in rows for d in r['peak_allocated_bytes']]) + ' (torch only)',
                 reserved=span([d / G for r in rows for d in r['peak_reserved_bytes']]) + ' (torch only)',
                 residency=f"{max(s['resident_bytes'] for s in stats) / (TOTAL_EXPERTS * EB):.2%} (resident bytes / all routed experts)")
    m['smi'] = '%.2f' % max(smi_max(j) for j in jobs)
    return m


rows, mem = [], []
for name, jobs in SYSTEMS:
    jobs = [j for j in jobs if passed(j)]
    if not jobs:
        continue
    reps = [(j, k, r) for j in jobs for k, r in repeats(j)]
    t = {key: [r[key] for _, _, r in reps] for key in ('TTFT', 'TPOT', 'E2E')}
    t['throughput'] = [tps(r) for _, _, r in reps]
    rows.append(dict(system=name, jobs=jobs, n=len(reps), samples=[dict(job=j, repeat=k, TTFT=r['TTFT'], TPOT=r['TPOT'], E2E=r['E2E'], TPS=tps(r)) for j, k, r in reps],
                     **{k: dict(median=st.median(v), min=min(v), max=max(v)) for k, v in t.items()}))
    mem.append(dict(system=name, **memory(name, jobs)))

(HERE / 'SUMMARY.json').write_text(json.dumps(dict(timing=rows, memory=mem), indent=2) + '\n')
print('| System | TPS ↑ | TTFT (s) ↓ | TPOT (s/token) ↓ | n |\n|---|---:|---:|---:|---:|')
for r in rows:
    f = lambda k, d: f"{r[k]['median']:.{d}f} [{r[k]['min']:.{d}f}, {r[k]['max']:.{d}f}]"
    print(f"| {r['system']} | {f('throughput', 2)} | {f('TTFT', 3)} | {f('TPOT', 4)} | {r['n']} |")
print('\n| System | Dense (GiB) | GPU-resident routed experts (GiB) | KV cache (GiB) | Runtime buffers / workspace (GiB) | Peak allocated (GiB) | Peak reserved (GiB) | nvidia-smi max (GiB) | Expert-cache residency |\n|---|---|---|---|---|---|---|---|---|')
for m in mem:
    print(f"| {m['system']} | {m['dense']} | {m['experts']} | {m['kv']} | {m['workspace']} | {m['alloc']} | {m['reserved']} | {m['smi']} | {m['residency']} |")
