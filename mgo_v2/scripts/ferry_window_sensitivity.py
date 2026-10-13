"""CPU-only Gate-history window (W) sensitivity replay for the FERRY appendix.

Inputs: a capture directory written with MGO_FERRY_WINDOW_PROBS=1 (routing_trace.npz
+ window_probs_rank{0..3}.npz). Gate traces for each W are rebuilt with the
production GateHistory class (mgo_v2.eviction) using the same prefill/decode
update calls as the live runtime; MAIN eviction is replayed with the same BR
quota, seed, tie-break and active pinning as main_eviction_history_replay.replay.
Run with CUDA_VISIBLE_DEVICES="".
"""
import argparse, csv, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from main_eviction_history_replay import replay as reference_replay  # noqa: E402
from mgo_v2.eviction import GateHistory  # noqa: E402

L, E, WORLD = 48, 128, 4
WINDOWS = (32, 64, 128, 256, 512, 1024)
CAPS = {30: [461, 461, 461, 460], 40: [615, 615, 614, 614], 60: [922, 922, 921, 921]}
HORIZONS = (1, 4, 16)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''): h.update(chunk)
    return h.hexdigest()


def load(capture):
    with np.load(capture / 'routing_trace.npz') as d:
        tr = {k: d[k] for k in d.files}
    ranks = []
    for r in range(WORLD):
        with np.load(capture / f'window_probs_rank{r}.npz') as d:
            ranks.append({k: d[k] for k in d.files})
    return tr, ranks


# ---------------------------------------------------------------- gate traces
def gate_trace(ranks, variant, order=(0, 1, 2, 3)):
    """variant: ('window', W) | ('balanced', W) | ('all', None) | ('current', None)."""
    kind, W = variant
    dec = [ranks[r]['decode'] for r in order]            # [events-48, B, E] each
    ev_dec = dec[0].shape[0]; events = ev_dec + L
    out = np.zeros((events, E), np.float32)
    if kind == 'window':
        h = GateHistory(L, E, W)
        for layer in range(L):
            tail = ranks[order[-1]]['prefill_tail'][layer]
            assert W <= len(tail)
            # Live prefill gathers only the exact global suffix; with W<=1024 it
            # lies entirely in the last rank of the concatenation order.
            h.update(layer, tail[-W:])
            out[layer] = (h.sums[layer] / max(1, len(h.rows[layer]))).astype(np.float32)
        for e in range(ev_dec):
            layer = e % L
            h.update_compiled(layer, np.concatenate([d[e] for d in dec]))
            out[e + L] = (h.sums[layer] / max(1, len(h.rows[layer]))).astype(np.float32)
    elif kind == 'balanced':
        assert W % WORLD == 0
        hs = [GateHistory(L, E, W // WORLD) for _ in range(WORLD)]
        def score(layer):
            s = sum(h.sums[layer] for h in hs); n = sum(len(h.rows[layer]) for h in hs)
            return (s / max(1, n)).astype(np.float32)
        for layer in range(L):
            for i, r in enumerate(order):
                hs[i].update(layer, ranks[r]['prefill_tail'][layer][-(W // WORLD):])
            out[layer] = score(layer)
        for e in range(ev_dec):
            layer = e % L
            for i in range(WORLD): hs[i].update_compiled(layer, dec[i][e])
            out[e + L] = score(layer)
    elif kind == 'all':
        s = sum(ranks[r]['prefill_sum'] for r in order).astype(np.float64)
        n = sum(ranks[r]['prefill_count'] for r in order).astype(np.float64)
        out[:L] = (s / n[:, None]).astype(np.float32)
        for e in range(ev_dec):
            layer = e % L; rows = np.concatenate([d[e] for d in dec]).astype(np.float64)
            s[layer] += rows.sum(0); n[layer] += len(rows)
            out[e + L] = (s[layer] / n[layer]).astype(np.float32)
    elif kind == 'current':
        s = sum(ranks[r]['prefill_sum'] for r in order); n = sum(ranks[r]['prefill_count'] for r in order)
        out[:L] = (s / n[:, None]).astype(np.float32)
        for e in range(ev_dec):
            out[e + L] = np.concatenate([d[e] for d in dec]).astype(np.float64).mean(0).astype(np.float32)
    else:
        raise ValueError(kind)
    return out


# ---------------------------------------------------------------- replay
@njit(cache=True)
def replay_tw(counts, gate_trace, capacities, seed=42, layers=48):
    """Copy of main_eviction_history_replay.replay(mode=0) + token-weighted hits."""
    events, experts = counts.shape; world = len(capacities); keys = layers * experts
    slots = np.full((world, int(capacities.max())), -1, np.int32)
    owner = np.full(keys, -1, np.int32); last = np.zeros(keys, np.int64)
    gates = np.zeros(keys, np.float32); ever = np.zeros(keys, np.bool_)
    metrics = np.zeros((events, 4), np.int64); tok = np.zeros((events, 2), np.int64)
    np.random.seed(seed)
    for event in range(events):
        layer = event % layers; active = counts[event] > 0
        gates[layer * experts:(layer + 1) * experts] = gate_trace[event]
        misses = np.empty(experts, np.int64); n = 0
        for e in range(experts):
            if active[e]:
                key = layer * experts + e
                tok[event, 1] += counts[event, e]
                if owner[key] >= 0:
                    metrics[event, 0] += 1; tok[event, 0] += counts[event, e]
                else:
                    misses[n] = e; n += 1
        metrics[event, 1] = n
        quota = np.empty(n, np.int64); at = 0
        for rank in range(world):
            for _ in range(n // world + int(rank < n % world)): quota[at] = rank; at += 1
        order = np.arange(n); np.random.shuffle(order); assignment = np.empty(n, np.int64)
        for j in range(n): assignment[order[j]] = quota[j]
        for j in range(n):
            expert = misses[j]; key = layer * experts + expert; rank = assignment[j]
            best = -1; best_score = np.inf; best_last = np.iinfo(np.int64).max; best_key = np.iinfo(np.int64).max
            for slot in range(capacities[rank]):
                victim = slots[rank, slot]
                if victim < 0: best = slot; break
                if victim // experts == layer and active[victim % experts]: continue
                score = float(gates[victim]); used = last[victim]
                if score < best_score or (score == best_score and (used < best_last or (used == best_last and victim < best_key))):
                    best = slot; best_score = score; best_last = used; best_key = victim
            if best < 0: raise ValueError('MAIN capacity cannot admit active experts under BR quota')
            victim = slots[rank, best]
            if victim >= 0: metrics[event, 2] += 1; owner[victim] = -1
            if ever[key]: metrics[event, 3] += 1
            ever[key] = True; slots[rank, best] = key; owner[key] = rank
        for e in range(experts):
            if active[e]: last[layer * experts + e] = event + 1
    return metrics, slots, tok


def spearman_rows(a, b):
    """Row-wise Spearman (average ranks); NaN when a row is constant."""
    from scipy.stats import rankdata
    ra = rankdata(a, axis=1); rb = rankdata(b, axis=1)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (ra * rb).sum(1) / den


def prediction(counts, gt):
    """Mean Spearman(score at decode event (s,l), demand of layer l over steps s+1..s+h)."""
    steps = counts.shape[0] // L
    c = counts.reshape(steps, L, E).astype(np.float64); g = gt.reshape(steps, L, E)
    cs = np.concatenate([np.zeros((1, L, E)), np.cumsum(c, 0)])
    out = {}
    for h in HORIZONS:
        s = np.arange(1, steps - h)          # decode steps with a full future window
        fut = cs[s + h + 1] - cs[s + 1]       # steps s+1 .. s+h
        rho = spearman_rows(g[s].reshape(-1, E), fut.reshape(-1, E))
        out[h] = float(np.nanmean(rho))
    return out


def evaluate(counts, gt, caps):
    rows = {}
    for pct in caps:
        m, _, tok = replay_tw(counts, gt, np.array(CAPS[pct], np.int32), 42)
        d = m[L:].sum(0); t = tok[L:].sum(0)
        rows[pct] = dict(hit=int(d[0]), miss=int(d[1]), eviction=int(d[2]), reload=int(d[3]),
                         hit_rate=d[0] / (d[0] + d[1]), token_hit_rate=t[0] / t[1])
    return rows


def label(variant):
    kind, W = variant
    return {'window': f'W{W}', 'balanced': f'balanced_W{W}', 'all': 'all_history', 'current': 'current_event'}[kind]


def job(args):
    capture, variant, order, caps = args
    tr, ranks = load(Path(capture)); counts = tr['counts']
    t0 = time.time(); gt = gate_trace(ranks, variant, order)
    res = evaluate(counts, gt, caps); pred = prediction(counts, gt)
    return dict(variant=label(variant), order=order, caps=res, pred=pred, seconds=time.time() - t0)


# ---------------------------------------------------------------- validity
def validate(capture, old_capture, scratch):
    tr, ranks = load(capture)
    receipt = json.loads((capture / 'trace_receipt.json').read_text())
    counts, gates, ref, final = tr['counts'], tr['gates'], tr['reference_metrics'], tr['final_slots']
    B = receipt['local_batch']; v = dict(local_batch=B)
    for r in range(WORLD):
        rr = json.loads((capture / f'window_probs_rank{r}.json').read_text())
        assert rr['status'] == 'PASS' and sha256(capture / rr['file']) == rr['file_sha256']
        v[f'rank{r}_probs_sha256'] = rr['file_sha256']
        assert ranks[r]['decode'].shape == (256 * L, B, E)
    v['routing_trace_sha256'] = sha256(capture / 'routing_trace.npz')
    v['content_sha256'] = receipt['content_sha256']
    # (a) raw rows are routing-consistent: argmax-top8 of probs vs counts (decode)
    allrows = np.concatenate([ranks[r]['decode'] for r in range(WORLD)], axis=1)  # [ev, 4B, E]
    top = np.argsort(-allrows, axis=2, kind='stable')[:, :, :8]
    rc = np.zeros((allrows.shape[0], E), np.int64)
    for e in range(allrows.shape[0]): rc[e] = np.bincount(top[e].ravel(), minlength=E)
    v['decode_top8_from_probs_matches_counts_events'] = int((rc == counts[L:]).all(1).sum())
    v['decode_events'] = int(allrows.shape[0])
    # (b) W=128 rebuilt from raw rows reproduces stored gates
    gt = gate_trace(ranks, ('window', 128))
    v['w128_gates_exact_equal'] = bool(np.array_equal(gt, gates))
    v['w128_gates_max_abs_diff'] = float(np.abs(gt.astype(np.float64) - gates).max())
    v['w128_gates_events_unequal'] = int((gt != gates).any(1).sum())
    # (c) gate replay parity at source capacity with rebuilt gates and with stored gates
    m, s, _ = replay_tw(counts, gt, np.array(receipt['capacities'], np.int32), 42)
    v['rebuilt_w128_replay_metrics_equal_live'] = bool(np.array_equal(m, ref))
    v['rebuilt_w128_replay_final_slots_equal_live'] = bool(np.array_equal(s, final))
    m0, s0, _, _ = reference_replay(counts, gates, np.array(receipt['capacities'], np.int32), 0, 42)
    v['reference_replay_stored_gates_parity'] = bool(np.array_equal(m0, ref) and np.array_equal(s0, final))
    v['replay_tw_equals_reference_replay'] = bool(np.array_equal(m0, replay_tw(counts, gates, np.array(receipt['capacities'], np.int32), 42)[0]))
    # (d) routing identity with the 2026-10-07 capture
    if old_capture is not None:
        with np.load(old_capture / 'routing_trace.npz') as d:
            oc, og, om = d['counts'], d['gates'], d['reference_metrics']
        v['old_trace'] = str(old_capture / 'routing_trace.npz')
        v['old_trace_sha256'] = sha256(old_capture / 'routing_trace.npz')
        v['counts_identical_to_old'] = bool(np.array_equal(oc, counts))
        v['counts_events_differing_from_old'] = int((oc != counts).any(1).sum())
        v['first_differing_event'] = int(np.flatnonzero((oc != counts).any(1))[0]) if not np.array_equal(oc, counts) else None
        v['gates_identical_to_old'] = bool(np.array_equal(og, gates))
        v['gates_max_abs_diff_vs_old'] = float(np.abs(og.astype(np.float64) - gates).max())
        v['reference_metrics_identical_to_old'] = bool(np.array_equal(om, ref))
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, nargs='+', required=True, help='capture dirs, B8 B16 B64 order')
    p.add_argument('--old', type=Path, nargs='+', help='matching 2026-10-07 capture dirs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only: set CUDA_VISIBLE_DEVICES=""'
    a.output.mkdir(parents=True, exist_ok=True)
    olds = a.old or [None] * len(a.capture)
    validity = [validate(c, o, a.output) for c, o in zip(a.capture, olds)]
    (a.output / 'VALIDITY.json').write_text(json.dumps(validity, indent=2) + '\n')
    print(json.dumps(validity, indent=2), flush=True)

    rng = np.random.default_rng(20261011); perms = []
    while len(perms) < 2:
        q = tuple(int(x) for x in rng.permutation(WORLD))
        if q not in ((0, 1, 2, 3), (3, 2, 1, 0)) and q not in perms: perms.append(q)
    orders = {'identity': (0, 1, 2, 3), 'reversed': (3, 2, 1, 0), 'random1': perms[0], 'random2': perms[1]}
    caps = tuple(CAPS)
    jobs = []
    for c in a.capture:
        for v in [('window', w) for w in WINDOWS] + [('all', None), ('current', None)]:
            jobs.append((str(c), v, orders['identity'], caps))
        for name in ('reversed', 'random1', 'random2'):
            for w in WINDOWS: jobs.append((str(c), ('window', w), orders[name], caps))
        for w in WINDOWS: jobs.append((str(c), ('balanced', w), orders['identity'], caps))
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(job, jobs))
    batch_of = {str(c): v['local_batch'] for c, v in zip(a.capture, validity)}
    inv = {v: k for k, v in orders.items()}
    flat = []
    for jb, r in zip(jobs, results):
        for pct, m in r['caps'].items():
            flat.append(dict(local_batch=batch_of[jb[0]], global_batch=4 * batch_of[jb[0]], capacity_pct=pct,
                             variant=r['variant'], order=inv[r['order']], order_ranks=''.join(map(str, r['order'])),
                             **m, spearman_h1=r['pred'][1], spearman_h4=r['pred'][4], spearman_h16=r['pred'][16]))
    (a.output / 'ALL_ROWS.json').write_text(json.dumps(dict(orders=orders, rows=flat), indent=2) + '\n')
    fields = ['local_batch', 'global_batch', 'capacity_pct', 'variant', 'order_ranks', 'hit', 'miss', 'eviction',
              'reload', 'hit_rate', 'token_hit_rate', 'spearman_h1', 'spearman_h4', 'spearman_h16']
    with (a.output / 'WINDOW_SENSITIVITY.csv').open('w') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore'); w.writeheader()
        for r in flat:
            if r['order'] == 'identity' and not r['variant'].startswith('balanced'): w.writerow(r)
    base = {(r['local_batch'], r['capacity_pct'], r['variant']): r for r in flat
            if r['order'] == 'identity' and r['variant'].startswith('W')}
    with (a.output / 'RANK_ORDER_ROBUSTNESS.csv').open('w') as f:
        fields2 = ['local_batch', 'global_batch', 'capacity_pct', 'window', 'arm', 'order_ranks', 'hit_rate',
                   'token_hit_rate', 'miss', 'spearman_h4', 'abs_diff_hit_rate_pp', 'abs_diff_token_hit_rate_pp', 'diff_miss']
        w = csv.DictWriter(f, fields2); w.writeheader()
        for r in flat:
            if r['variant'] in ('all_history', 'current_event'): continue
            win = r['variant'].replace('balanced_', '')
            arm = 'rank_balanced' if r['variant'].startswith('balanced') else r['order']
            b = base[(r['local_batch'], r['capacity_pct'], win)]
            w.writerow(dict(local_batch=r['local_batch'], global_batch=r['global_batch'], capacity_pct=r['capacity_pct'],
                            window=int(win[1:]), arm=arm, order_ranks=r['order_ranks'], hit_rate=r['hit_rate'],
                            token_hit_rate=r['token_hit_rate'], miss=r['miss'], spearman_h4=r['spearman_h4'],
                            abs_diff_hit_rate_pp=100 * abs(r['hit_rate'] - b['hit_rate']),
                            abs_diff_token_hit_rate_pp=100 * abs(r['token_hit_rate'] - b['token_hit_rate']),
                            diff_miss=r['miss'] - b['miss']))
    print('done', len(flat), 'rows', flush=True)


if __name__ == '__main__':
    main()
