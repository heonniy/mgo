"""Audit token-path agreement without checking raw prompt or token IDs into Git."""

import itertools
import json
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
REPORT = PKG / 'experiments/deepseek_cache_ablation_20261009'
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
LABELS = {'ours': 'main_OURS', 'infinity': 'MoE-Infinity (repaired)',
          'deepspeed': 'DeepSpeed ZeRO-Inference', 'llama': 'llama.cpp balanced'}


def tokens(case, repeat):
    path = JOBS / case['selected_attempt']
    if case['system'] in ('ours', 'deepspeed'):
        rows = [json.loads((path / f'repeat{repeat}_rank{rank}.json').read_text())
                for rank in range(4)]
        ids = list(itertools.chain.from_iterable(row['request_ids'] for row in rows))
        generated = list(itertools.chain.from_iterable(row['tokens'] for row in rows))
    else:
        row = json.loads((path / f'repeat{repeat}.json').read_text())
        ids, generated = row['request_ids'], row['tokens']
    assert len(ids) == len(generated) == 64
    assert all(len(row) == 64 for row in generated)
    return ids, generated


def agreement(left, right):
    assert left[0] == right[0], 'request membership/order differs'
    a, b = left[1], right[1]
    return dict(first=sum(x[0] == y[0] for x, y in zip(a, b)),
                full=sum(x == y for x, y in zip(a, b)),
                positions=sum(sum(i == j for i, j in zip(x, y))
                              for x, y in zip(a, b)))


def main():
    progress = json.loads((REPORT / 'PROGRESS.json').read_text())
    cases = {(case['cache_percent'], case['system']): case for case in progress['cases']
             if case['status'] == 'PASS'}
    lines = ['# Decode token-path stability', '',
             'All cells use the same 64 frozen target requests and start from an empty dynamic '
             'cache after warmup. Agreement is descriptive; it does not change the primary '
             'greedy TTFT/TPOT/E2E measurements. Raw request IDs and generated token IDs remain '
             'outside Git. Each comparison has 64 requests and 4,096 generated token positions.', '',
             '## Within a cache setting', '',
             '| Cache | System | Full-64 request agreement across repeats 1–2 / 1–3 / 2–3 | '
             'First-token agreement across repeats 1–2 / 1–3 / 2–3 |',
             '|---:|---|---|---|']
    loaded = {}
    for cache in (20, 30, 40, 50):
        for system in LABELS:
            case = cases.get((cache, system))
            if case is None:
                continue
            values = [tokens(case, repeat) for repeat in (1, 2, 3)]
            loaded[cache, system] = values
            checks = [agreement(values[a], values[b]) for a, b in ((0, 1), (0, 2), (1, 2))]
            full = ' / '.join(f'{item["full"]}/64' for item in checks)
            first = ' / '.join(f'{item["first"]}/64' for item in checks)
            lines.append(f'| {cache}% | {LABELS[system]} | {full} | {first} |')
    lines.extend(['', '## C20 reference versus other cache sizes', '',
                  'Compare target repeat 2 at each cache size; first-token agreement, complete '
                  '64-token request agreement, and agreement over all token positions are separate.', '',
                  '| System | Comparison | First token | Full 64 tokens | Token positions |',
                  '|---|---|---:|---:|---:|'])
    for system in LABELS:
        reference = loaded.get((20, system))
        if reference is None:
            continue
        for cache in (30, 40, 50):
            other = loaded.get((cache, system))
            if other is None:
                continue
            item = agreement(reference[1], other[1])
            lines.append(f'| {LABELS[system]} | C20→C{cache} | '
                         f'{item["first"]}/64 | {item["full"]}/64 | '
                         f'{item["positions"]}/4096 |')
    lines.extend(['', 'Capacity changes can alter BF16 execution order and, after a token '
                  'diverges, later routing demand. Therefore the measured end-to-end results '
                  'describe actual greedy runs at each capacity; traffic differences do not '
                  'isolate capacity under a bit-identical future decode trajectory.'])
    (REPORT / 'TOKEN_STABILITY.md').write_text('\n'.join(lines) + '\n')
    print(f'token audit: {len(loaded)} completed system-cache cells')


if __name__ == '__main__':
    main()
