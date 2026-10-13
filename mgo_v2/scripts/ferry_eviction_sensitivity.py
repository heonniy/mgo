"""FERRY appendix: CPU-only expert-cache eviction sensitivity sweep.

Replays frozen routing traces (Qwen3-30B-A3B R4 B8/B16/B64; DeepSeek-V2-Lite D1
B16) under MAIN-only BR admission (seed 42) for a grid of cache budgets.

Policies (all keyed per (layer, expert); active experts of the current event
are pinned; ties broken by score, then last-use, then key):
  gate-score       W128 gate-score vector (lowest evicted)
  LFU-reset        +1 per (decode step, layer, expert) event; freq/last cleared on eviction
  LFU-cumulative   same counter, retained across eviction/re-admission
  LRU-reset        last-use event index (cleared on eviction)
  LRU-cumulative   last-use retained (provably identical decisions to LRU-reset)
  random           uniform victim among evictable slots (own xorshift RNG, seed 1234;
                   BR np.random stream untouched)
  OPT-BR (oracle)  Belady farthest-next-use victim under the same BR admission
  OPT-pooled (oracle) Belady on one pooled cache of the same total size
                   (no rank partition): upper bound on distinct hits of any
                   demand-admission policy with that many slots.

Modes 0-4 are cross-checked bit-exactly (per-event hit/miss/evict/reload and
final slots) against mgo_v2/scripts/main_eviction_history_replay.replay at
every feasible cell. Gate-score is checked against the source capture's
reference_metrics/final_slots at the source capacity (Qwen C30) and against
the prior DeepSeek RESULTS.json at C30/C60.
NO GPU: run with CUDA_VISIBLE_DEVICES="".
"""
import argparse, csv, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
from numba import njit

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from main_eviction_history_replay import replay as reference_replay  # noqa: E402

ROOT = SCRIPTS.parent
OUT = ROOT / 'experiments/ferry_appendix_sensitivity_20261011/eviction'
QWEN = Path('/home/hwlee/mgo-results/headline_r4_20261007')
POLICIES = ('gate-score', 'LFU-reset', 'LFU-cumulative', 'LRU-reset', 'LRU-cumulative',
            'random', 'OPT-BR (oracle)', 'OPT-pooled (oracle)')
PCTS = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)
NEVER = np.iinfo(np.int64).max // 4


@njit(cache=True)
def replay2(counts, gate_trace, nextocc, capacities, mode, seed, layers, rng_seed):
    events, experts = counts.shape
    world = len(capacities); keys = layers * experts
    slots = np.full((world, int(capacities.max())), -1, np.int32)
    owner = np.full(keys, -1, np.int32); freq = np.zeros(keys, np.int64)
    last = np.zeros(keys, np.int64); nxt = np.zeros(keys, np.int64)
    gates = np.zeros(keys, np.float32); ever = np.zeros(keys, np.bool_)
    metrics = np.zeros((events, 5), np.int64)  # hit, miss, eviction, reload, token_hit
    np.random.seed(seed)
    rs = np.uint64(rng_seed)
    for event in range(events):
        layer = event % layers; active = counts[event] > 0
        gates[layer * experts:(layer + 1) * experts] = gate_trace[event]
        misses = np.empty(experts, np.int64); n = 0
        for e in range(experts):
            if active[e]:
                key = layer * experts + e
                if owner[key] >= 0:
                    metrics[event, 0] += 1; metrics[event, 4] += counts[event, e]
                else:
                    misses[n] = e; n += 1
        metrics[event, 1] = n
        quota = np.empty(n, np.int64); at = 0
        for rank in range(world):
            for _ in range(n // world + int(rank < n % world)):
                quota[at] = rank; at += 1
        order = np.arange(n); np.random.shuffle(order); assignment = np.empty(n, np.int64)
        for j in range(n):
            assignment[order[j]] = quota[j]
        for j in range(n):
            expert = misses[j]; key = layer * experts + expert; rank = assignment[j]
            best = -1; best_score = np.inf
            best_last = np.iinfo(np.int64).max; best_key = np.iinfo(np.int64).max
            for slot in range(capacities[rank]):
                victim = slots[rank, slot]
                if victim < 0:
                    best = slot; break
                if victim // experts == layer and active[victim % experts]:
                    continue
                if mode == 0:
                    score = float(gates[victim])
                elif mode <= 2:
                    score = float(freq[victim])
                elif mode <= 4:
                    score = 0.
                elif mode == 5:
                    rs ^= rs << np.uint64(13); rs ^= rs >> np.uint64(7); rs ^= rs << np.uint64(17)
                    score = float(rs >> np.uint64(11)) / 9007199254740992.0
                else:
                    score = -float(nxt[victim])
                used = last[victim]
                if score < best_score or (score == best_score and (used < best_last or (used == best_last and victim < best_key))):
                    best = slot; best_score = score; best_last = used; best_key = victim
            if best < 0:
                raise ValueError('MAIN capacity cannot admit active experts under BR quota')
            victim = slots[rank, best]
            if victim >= 0:
                metrics[event, 2] += 1; owner[victim] = -1
                if mode == 1 or mode == 3:
                    freq[victim] = 0; last[victim] = 0
            if ever[key]:
                metrics[event, 3] += 1
            ever[key] = True; slots[rank, best] = key; owner[key] = rank
        for e in range(experts):
            if active[e]:
                key = layer * experts + e; freq[key] += 1; last[key] = event + 1
                nxt[key] = nextocc[event, e]
    return metrics, slots


def next_occurrence(counts, layers):
    events, experts = counts.shape
    out = np.full((events, experts), NEVER, np.int64)
    seen = np.full((layers, experts), NEVER, np.int64)
    for event in range(events - 1, -1, -1):
        layer = event % layers
        out[event] = seen[layer]
        seen[layer][counts[event] > 0] = event
    return out


def split(total, world, reserve):
    return np.array([total // world + (r < total % world) - reserve for r in range(world)], np.int32)


# ---------------------------------------------------------------- loaders
def load_qwen(batch):
    d = QWEN / f'main_eviction_B{batch}_L256_H256_v1'
    receipt = json.loads((d / 'trace_receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['decode_forwards'] == 256 and receipt['prefill_forwards'] == 1
    assert not receipt['prefetch'] and receipt['all_rank_agreement'] and receipt['local_batch'] == batch
    assert receipt['eviction'] == 'gate-score' and receipt['policy'] == 'BR' and receipt['seed'] == 42
    trace = d / 'routing_trace.npz'
    with np.load(trace, allow_pickle=False) as z:
        counts = z['counts']; gates = z['gates']; ref = z['reference_metrics']; final = z['final_slots']
    assert counts.shape == (257 * 48, 128) and gates.shape == counts.shape and counts.dtype == np.int32
    assert np.all(counts >= 0) and np.isfinite(gates).all()
    assert np.all(counts[48:].sum(1) == batch * 4 * 8)
    content = hashlib.sha256(counts.tobytes() + gates.tobytes() + ref.tobytes()).hexdigest()
    assert content == receipt['content_sha256']
    return dict(model='Qwen3-30B-A3B', batch=batch, layers=48, experts=128, steps=256, world=4, reserve=0,
                counts=counts, gates=gates, reference=ref, final_slots=final,
                source_capacities=receipt['capacities'], trace=str(trace),
                file_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), content_sha256=content,
                topk=8, receipt=str(d / 'trace_receipt.json'))


def load_deepseek():
    import replay_deepseek_eviction_hit as ds
    files, hashes = ds.validated_files()
    counts, gates = ds.pack(files)
    prior = json.loads((ROOT / 'experiments/deepseek_eviction_hit_20261009/RESULTS.json').read_text())
    assert prior['source_sha256'] == hashes
    return dict(model='DeepSeek-V2-Lite', batch=16, layers=26, experts=64, steps=63, world=4, reserve=2,
                counts=counts, gates=gates, reference=None, final_slots=None, source_capacities=None,
                trace=[str(f) for f in files], file_sha256=hashes,
                content_sha256=hashlib.sha256(counts.tobytes() + gates.tobytes()).hexdigest(),
                topk=6, prior=prior, receipt=str(ds.REFERENCE / 'status.json'))


# ---------------------------------------------------------------- one cell
def run_cell(src, pct, nextocc):
    L, E, W = src['layers'], src['experts'], src['world']
    counts, gates = src['counts'], src['gates']
    total = L * E * pct // 100
    caps = split(total, W, src['reserve'])
    dec = slice(L, None)
    token_total = int(counts[dec].sum())
    distinct_total = int((counts[dec] > 0).sum())
    rows, checks = [], []
    lru = None
    for mode, name in enumerate(POLICIES):
        c = np.array([caps.sum()], np.int32) if mode == 7 else caps
        m = 6 if mode == 7 else mode
        base = dict(model=src['model'], batch=src['batch'], capacity_frac=pct / 100, capacity_pct=pct,
                    slots=int(c.sum()), capacities=' '.join(map(str, c)), policy=name)
        if np.any(c <= 0):
            rows.append(dict(base, feasible=False, note='non-positive per-rank MAIN capacity')); continue
        try:
            metrics, slots = replay2(counts, gates, nextocc, c, m, 42, L, 1234)
        except Exception as exc:  # BR quota cannot place all active experts
            rows.append(dict(base, feasible=False, note=f'{type(exc).__name__}: {exc}')); continue
        # conservation
        assert np.array_equal(metrics[:, :2].sum(1), (counts > 0).sum(1))
        assert np.all(metrics[:, 3] <= metrics[:, 1]) and np.all(metrics[:, 2] <= metrics[:, 1])
        assert int(metrics[:, 1].sum() - metrics[:, 2].sum()) == int((slots >= 0).sum())
        assert np.all(metrics[:, 4] <= counts.sum(1))
        if mode <= 4:  # bit-exact vs the frozen study kernel
            rm, rs, _, _ = reference_replay(counts, gates, caps, mode, 42, L)
            assert np.array_equal(rm, metrics[:, :4]) and np.array_equal(rs, slots), (name, pct)
            checks.append(f'{name}: matches main_eviction_history_replay')
        if mode == 0 and src['source_capacities'] is not None and list(caps) == list(src['source_capacities']):
            assert np.array_equal(metrics[:, :4], src['reference']) and np.array_equal(slots, src['final_slots'])
            checks.append('gate-score: matches source capture reference_metrics and final_slots')
        if mode == 3:
            lru = (metrics, slots)
        if mode == 4:
            assert np.array_equal(lru[0], metrics) and np.array_equal(lru[1], slots)
            checks.append('LRU-reset == LRU-cumulative')
        d = metrics[dec].sum(0)
        assert d[0] + d[1] == distinct_total
        rows.append(dict(base, feasible=True, decode_hit=int(d[0]), decode_miss=int(d[1]),
                         hit_rate=d[0] / distinct_total, token_hit=int(d[4]), token_total=token_total,
                         token_weighted_hit_rate=d[4] / token_total, evictions=int(d[2]), reloads=int(d[3]),
                         prefill_hit=int(metrics[:L, 0].sum()), prefill_miss=int(metrics[:L, 1].sum())))
    return rows, checks


def worker(args):
    key, pct = args
    src = SOURCES[key]
    return key, pct, run_cell(src, pct, NEXTOCC[key])


SOURCES, NEXTOCC = {}, {}


def init(keys):
    for k in keys:
        SOURCES[k] = load_deepseek() if k == 'ds16' else load_qwen(int(k[1:]))
        NEXTOCC[k] = next_occurrence(SOURCES[k]['counts'], SOURCES[k]['layers'])
    # warm numba
    s = SOURCES[keys[0]]
    replay2(s['counts'][:2 * s['layers']], s['gates'][:2 * s['layers']], NEXTOCC[keys[0]][:2 * s['layers']],
            np.array([s['layers'] * s['experts']] * 4, np.int32), 0, 42, s['layers'], 1)


def working_set(src):
    L = src['layers']; a = (src['counts'] > 0).reshape(-1, L, src['experts'])[1:]
    per_step = a.sum((1, 2)); always = a.all(0).sum()
    return dict(per_step_distinct_min=int(per_step.min()), per_step_distinct_mean=float(per_step.mean()),
                per_step_distinct_max=int(per_step.max()), active_every_decode_step=int(always),
                touched_in_decode=int(a.any(0).sum()), total_experts=L * src['experts'],
                per_layer_active_max=int(a.sum(2).max()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--jobs', type=int, default=24)
    p.add_argument('--sources', nargs='+', default=['q8', 'q16', 'q64', 'ds16'])
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES', None) == '', 'run with CUDA_VISIBLE_DEVICES=""'
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    init(a.sources)
    tasks = [(k, pct) for k in a.sources for pct in PCTS]
    results = {}
    with ProcessPoolExecutor(a.jobs, initializer=init, initargs=(a.sources,)) as ex:
        for key, pct, res in ex.map(worker, tasks):
            results[(key, pct)] = res
            print(key, pct, 'done', flush=True)
    rows, checks = [], {}
    for k in a.sources:
        for pct in PCTS:
            r, c = results[(k, pct)]; rows += r; checks[f'{k}/C{pct}'] = c
    # DeepSeek parity with prior RESULTS.json
    if 'ds16' in a.sources:
        prior = SOURCES['ds16']['prior']
        for blk in prior['results']:
            for pol in blk['policies']:
                mine = next(r for r in rows if r['model'] == 'DeepSeek-V2-Lite' and r['capacity_pct'] == blk['cache_percent'] and r['policy'] == pol['policy'])
                assert mine['capacities'] == ' '.join(map(str, blk['capacities']))
                for f in ('hit', 'miss', 'eviction', 'reload'):
                    assert mine[{'hit': 'decode_hit', 'miss': 'decode_miss', 'eviction': 'evictions', 'reload': 'reloads'}[f]] == pol['decode'][f]
            checks[f"ds16/C{blk['cache_percent']}"].append('all 5 policies match deepseek_eviction_hit_20261009/RESULTS.json')
    cols = ['model', 'batch', 'capacity_frac', 'slots', 'policy', 'decode_hit', 'decode_miss', 'hit_rate',
            'token_weighted_hit_rate', 'evictions', 'reloads', 'token_hit', 'token_total', 'capacity_pct',
            'capacities', 'feasible', 'prefill_hit', 'prefill_miss', 'note']
    with (OUT / 'EVICTION_SENSITIVITY.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, cols); w.writeheader()
        for r in rows:
            w.writerow({c: (f'{r[c]:.6f}' if isinstance(r.get(c), float) and c != 'capacity_frac' else r.get(c, '')) for c in cols})
    prov = dict(status='PASS', script=str(Path(__file__).resolve()),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_kernel=str(SCRIPTS / 'main_eviction_history_replay.py'),
                reference_kernel_sha256=hashlib.sha256((SCRIPTS / 'main_eviction_history_replay.py').read_bytes()).hexdigest(),
                deepseek_packer=str(SCRIPTS / 'replay_deepseek_eviction_hit.py'),
                deepseek_packer_sha256=hashlib.sha256((SCRIPTS / 'replay_deepseek_eviction_hit.py').read_bytes()).hexdigest(),
                command=' '.join(sys.argv), python=sys.executable, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                percents=PCTS, policies=POLICIES, br_seed=42, random_seed=1234,
                sources={k: dict(model=s['model'], batch=s['batch'], trace=s['trace'], file_sha256=s['file_sha256'],
                                 content_sha256=s['content_sha256'], receipt=s['receipt'],
                                 working_set=working_set(s)) for k, s in SOURCES.items()},
                checks=checks, wall_seconds=time.time() - t0)
    (OUT / 'PROVENANCE.json').write_text(json.dumps(prov, indent=2) + '\n')
    print(json.dumps({k: v['working_set'] for k, v in prov['sources'].items()}, indent=1))


if __name__ == '__main__':
    main()
