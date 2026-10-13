"""FERRY Appendix C: why Gate-score eviction, and how to pick its window W.

CPU-only replay of raw router-probability captures (MGO_FERRY_WINDOW_PROBS=1;
routing_trace.npz + window_probs_rank{0..3}.npz) under MAIN-only BR admission
(seed 42, active pinning, score -> last-use -> key tie-break), i.e. exactly the
kernel of main_eviction_history_replay / ferry_eviction_sensitivity.replay2,
extended with eviction logging. Only Step 4 (victim choice) changes.

Score-trace policies (rolling W-token window per layer, production GateHistory):
  full-gate      mean router probability over ALL experts (FERRY, W=128)
  selected-gate  router probability kept only for each token's top-k experts
  sel-count      top-k selection indicator (= windowed selection count)
Stateful baselines: LRU, LFU-cumulative (distinct events), random, OPT-BR (Belady).

Run with CUDA_VISIBLE_DEVICES="".
"""
import argparse, csv, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from ferry_window_sensitivity import load, gate_trace, spearman_rows  # noqa: E402
from ferry_eviction_sensitivity import replay2, next_occurrence, NEVER  # noqa: E402

L, E, WORLD, TOPK = 48, 128, 4, 8
EXPERT_BYTES = 9437184  # Qwen3-30B-A3B BF16 expert (3 x 2048 x 768 x 2 B)
WINDOWS = (8, 16, 32, 64, 128, 256, 512, 1024)
PCTS = (20, 30, 40, 60)
HORIZONS = (256, 64)  # decode forwards replayed; 64 = paper O64
SCORE_POLICIES = ('full-gate', 'selected-gate', 'sel-count', 'count+gate-tiebreak')
# count+gate-tiebreak (diagnostic): selection count, ties broken by full gate.
# Counts are multiples of 1/W >= 1/1024 and gates <= 1, so 1e-4 * gate is lexicographic.
BASELINES = {'LFU-cumulative': 2, 'LRU': 3, 'random': 5, 'OPT-BR (oracle)': 6}
PRED_H = (1, 4, 16)


def caps(pct):
    total = L * E * pct // 100
    return np.array([total // WORLD + (r < total % WORLD) for r in range(WORLD)], np.int32)


@njit(cache=True)
def replay_log(counts, gate_trace, nextocc, capacities, mode, seed, layers, rng_seed):
    """replay2 + logs. evict_log: (event, victim key, victim next use, tie count)."""
    events, experts = counts.shape
    world = len(capacities); keys = layers * experts
    slots = np.full((world, int(capacities.max())), -1, np.int32)
    owner = np.full(keys, -1, np.int32); freq = np.zeros(keys, np.int64)
    last = np.zeros(keys, np.int64); nxt = np.zeros(keys, np.int64)
    gates = np.zeros(keys, np.float32); ever = np.zeros(keys, np.bool_)
    evicted_at = np.full(keys, -1, np.int64)
    metrics = np.zeros((events, 6), np.int64)  # hit, miss, eviction, reload, token_hit, token_total
    elog = np.zeros((events * experts, 4), np.int64); ne = 0
    rlog = np.zeros((events * experts, 2), np.int64); nr = 0  # (event, gap since eviction)
    np.random.seed(seed)
    rs = np.uint64(rng_seed)
    for event in range(events):
        layer = event % layers; active = counts[event] > 0
        gates[layer * experts:(layer + 1) * experts] = gate_trace[event]
        misses = np.empty(experts, np.int64); n = 0
        for e in range(experts):
            if active[e]:
                key = layer * experts + e
                metrics[event, 5] += counts[event, e]
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
            ties = 0
            for slot in range(capacities[rank]):
                victim = slots[rank, slot]
                if victim < 0:
                    best = slot; ties = -1; break
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
                if score < best_score:
                    ties = 1
                elif score == best_score:
                    ties += 1
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
                elog[ne, 0] = event; elog[ne, 1] = victim; elog[ne, 2] = nxt[victim]; elog[ne, 3] = ties; ne += 1
                evicted_at[victim] = event
            if ever[key]:
                metrics[event, 3] += 1
                if evicted_at[key] >= 0:
                    rlog[nr, 0] = event; rlog[nr, 1] = event - evicted_at[key]; nr += 1
            ever[key] = True; slots[rank, best] = key; owner[key] = rank
        for e in range(experts):
            if active[e]:
                key = layer * experts + e; freq[key] += 1; last[key] = event + 1
                nxt[key] = nextocc[event, e]
    return metrics, slots, elog[:ne], rlog[:nr]


# ---------------------------------------------------------------- score traces
def topk_mask(rows):
    """Top-k indicator per token row; stable argsort as in the validity check."""
    top = np.argsort(-rows, axis=-1, kind='stable')[..., :TOPK]
    m = np.zeros(rows.shape, np.float32)
    np.put_along_axis(m, top, 1.0, axis=-1)
    return m


def transformed(ranks, policy):
    if policy == 'full-gate':
        return ranks
    out = []
    for r in ranks:
        d = {}
        for k in ('decode', 'prefill_tail'):
            m = topk_mask(r[k])
            d[k] = m if policy == 'sel-count' else (r[k] * m).astype(np.float32)
        out.append(d)
    return out


def score_trace(ranks, policy, variant):
    if policy == 'count+gate-tiebreak':
        return (gate_trace(transformed(ranks, 'sel-count'), variant).astype(np.float64)
                + 1e-4 * gate_trace(ranks, variant)).astype(np.float32)
    return gate_trace(transformed(ranks, policy), variant)


def recency_lfu_traces(counts):
    """Per-event score vectors of LRU (last active step) and LFU-cumulative."""
    steps = counts.shape[0] // L
    act = (counts.reshape(steps, L, E) > 0)
    lfu = np.cumsum(act, 0).astype(np.float32)
    idx = np.where(act, np.arange(steps)[:, None, None], -1)
    lru = np.maximum.accumulate(idx, 0).astype(np.float32)
    return lru.reshape(-1, E), lfu.reshape(-1, E)


# ---------------------------------------------------------------- analysis
def prediction(counts, score, sel_trace):
    """Spearman vs future demand; AUC of next-step activity overall and among
    experts with zero top-k selections in the window ('unselected')."""
    from scipy.stats import rankdata
    steps = counts.shape[0] // L
    c = counts.reshape(steps, L, E).astype(np.float64); g = score.reshape(steps, L, E)
    cs = np.concatenate([np.zeros((1, L, E)), np.cumsum(c, 0)])
    out = {}
    for h in PRED_H:
        s = np.arange(1, steps - h)
        fut = cs[s + h + 1] - cs[s + 1]
        out[f'spearman_h{h}'] = float(np.nanmean(spearman_rows(g[s].reshape(-1, E), fut.reshape(-1, E))))
    s = np.arange(1, steps - 1)
    y = (c[s + 1] > 0).reshape(-1, E); x = g[s].reshape(-1, E)
    cold = (sel_trace.reshape(steps, L, E)[s] == 0).reshape(-1, E)
    def auc(mask):
        vals = []
        for i in range(len(x)):
            m = mask[i]; yy = y[i][m]; P = int(yy.sum()); N = int(len(yy) - P)
            if P == 0 or N == 0: continue
            r = rankdata(x[i][m]); vals.append((r[yy].sum() - P * (P + 1) / 2) / (P * N))
        return float(np.mean(vals)) if vals else float('nan')
    out['auc_next_active'] = auc(np.ones_like(y))
    out['auc_next_active_unselected'] = auc(cold)
    out['unselected_share_of_next_active'] = float((y & cold).sum() / y.sum())
    return out


def summarize(metrics, elog, rlog, start_event):
    d = metrics[start_event:].sum(0)
    dec = elog[elog[:, 0] >= start_event]; rl = rlog[rlog[:, 0] >= start_event, 1]  # decode reloads only
    dist = np.where(dec[:, 2] >= NEVER, np.inf, (dec[:, 2] - dec[:, 0]) / L) if len(dec) else np.zeros(0)
    ties = dec[:, 3][dec[:, 3] > 0] if len(dec) else np.zeros(0)
    q = lambda a, p: float(np.quantile(a, p)) if len(a) else float('nan')
    return dict(
        hit=int(d[0]), miss=int(d[1]), eviction=int(d[2]), reload=int(d[3]),
        hit_rate=d[0] / (d[0] + d[1]), token_hit_rate=d[4] / d[5],
        h2d_fetches=int(d[1]), h2d_gib=d[1] * EXPERT_BYTES / 2**30, refetch=int(d[3]),
        victim_reuse_median_steps=q(dist[np.isfinite(dist)], .5) if len(dist) else float('nan'),
        victim_reused_within_1step=float((dist <= 1).mean()) if len(dist) else float('nan'),
        victim_reused_within_4steps=float((dist <= 4).mean()) if len(dist) else float('nan'),
        victim_never_reused=float(np.isinf(dist).mean()) if len(dist) else float('nan'),
        refetch_within_1step=float((rl <= L).mean()) if len(rl) else float('nan'),
        tie_median=q(ties, .5), tie_mean=float(ties.mean()) if len(ties) else float('nan'),
        tie_gt1_frac=float((ties > 1).mean()) if len(ties) else float('nan'),
        decode_evictions_logged=int(len(dec)))


def reuse_hist(elog, start_event):
    dec = elog[elog[:, 0] >= start_event]
    dist = np.where(dec[:, 2] >= NEVER, 10**6, (dec[:, 2] - dec[:, 0]))
    edges = np.array([0, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 10**5, 10**7])
    return dict(edges_events=edges.tolist(), counts=np.histogram(dist, edges)[0].tolist())


def run_cell(counts, gt, horizon, pct, mode, check=None):
    ev = (horizon + 1) * L
    cc = np.ascontiguousarray(counts[:ev]); g = np.ascontiguousarray(gt[:ev])
    nxt = next_occurrence(cc, L); cap = caps(pct)
    m, s, elog, rlog = replay_log(cc, g, nxt, cap, mode, 42, L, 1234)
    m2, s2 = replay2(cc, g, nxt, cap, mode, 42, L, 1234)
    assert np.array_equal(m[:, :5], m2) and np.array_equal(s, s2), 'logging kernel diverged from replay2'
    assert np.array_equal(m[:, :2].sum(1), (cc > 0).sum(1))
    if check is not None:
        assert np.array_equal(m[:, :4], check[0][:ev]) and np.array_equal(s, check[1]), 'source parity'
    return m, elog, rlog


def job(args):
    capture, policy, variant, tag = args
    t0 = time.time()
    tr, ranks = load(Path(capture)); counts = tr['counts']
    receipt = json.loads((Path(capture) / 'trace_receipt.json').read_text())
    if policy in SCORE_POLICIES:
        gt = score_trace(ranks, policy, variant); mode = 0
        W = variant[1] if variant[0] == 'window' else 128
        sel = score_trace(ranks, 'sel-count', ('window', W))
    else:
        gt = tr['gates']; mode = BASELINES[policy]
        sel = score_trace(ranks, 'sel-count', ('window', 128))
    rows = []; hists = {}
    for horizon in HORIZONS:
        for pct in PCTS:
            check = None
            if (policy == 'full-gate' and variant == ('window', 128) and horizon == 256
                    and caps(pct).tolist() == receipt['capacities']):
                check = (tr['reference_metrics'], tr['final_slots'])
            m, elog, rlog = run_cell(counts, gt, horizon, pct, mode, check)
            r = dict(tag=tag, local_batch=receipt['local_batch'], policy=policy,
                     variant=f'W{variant[1]}' if variant[0] == 'window' else variant[0],
                     window=variant[1] if variant[0] == 'window' else None,
                     horizon=horizon, capacity_pct=pct, source_parity=check is not None,
                     **summarize(m, elog, rlog, L))
            rows.append(r)
            if pct in (30, 60): hists[f'{horizon}_C{pct}'] = reuse_hist(elog, L)
    if policy in SCORE_POLICIES:
        score = gt
    else:
        lru, lfu = recency_lfu_traces(counts)
        score = {'LRU': lru, 'LFU-cumulative': lfu}.get(policy)
    pred = prediction(counts, score, sel) if score is not None else {}
    for r in rows: r.update(pred)
    return dict(rows=rows, hists=hists, seconds=time.time() - t0)


def jobs_for(capture, tag):
    out = []
    for policy in SCORE_POLICIES:
        for W in WINDOWS:
            out.append((capture, policy, ('window', W), tag))
    out.append((capture, 'full-gate', ('all', None), tag))
    for b in BASELINES:
        out.append((capture, b, ('n/a', None), tag))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, nargs='+', required=True)
    p.add_argument('--tag', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--policies', nargs='*')
    a = p.parse_args(); assert len(a.capture) == len(a.tag)
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = [j for c, t in zip(a.capture, a.tag) for j in jobs_for(str(c), t)]
    if a.policies: jobs = [j for j in jobs if j[1] in a.policies]
    # Big batches first for load balance.
    jobs.sort(key=lambda j: -json.loads((Path(j[0]) / 'trace_receipt.json').read_text())['local_batch'])
    t0 = time.time(); results = []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(job, jobs):
            results.append(res)
            r = res['rows'][0]
            print(f"{r['tag']} B{r['local_batch']} {r['policy']} {r['variant']} {res['seconds']:.0f}s", flush=True)
    rows = [r for res in results for r in res['rows']]
    keys = list(rows[0])
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with (a.output / 'POLICY_ROWS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    hist = [dict(tag=res['rows'][0]['tag'], local_batch=res['rows'][0]['local_batch'], policy=res['rows'][0]['policy'],
                 variant=res['rows'][0]['variant'], hist=res['hists']) for res in results]
    (a.output / 'REUSE_HIST.json').write_text(json.dumps(hist, indent=1))
    (a.output / 'RUN.json').write_text(json.dumps(dict(
        captures=[str(c) for c in a.capture], tags=a.tag, wall_seconds=time.time() - t0, jobs=len(jobs),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parity=sum(r['source_parity'] for r in rows)), indent=2))
    print('rows', len(rows), 'parity cells', sum(r['source_parity'] for r in rows), f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
