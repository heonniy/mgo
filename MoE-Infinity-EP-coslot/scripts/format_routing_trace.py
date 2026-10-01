"""Pretty-print the coslot routing trace (requests + per-layer plan/cache).

Usage:
  python3 scripts/format_routing_trace.py <results_dir> [--run-tag TAG]
      [--step N] [--layers a,b,c] [--max-fetch N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load_requests(d, tag):
    reqs = {}
    for f in sorted(glob.glob(f"{d}/{tag}_requests_rank*.json")):
        m = json.load(open(f))
        reqs[m["rank"]] = m
    return reqs


def load_global(d, tag):
    # rank0's JSONL holds the global plan/routing/cache (deterministic, same on
    # every rank).
    f = f"{d}/{tag}_routing_rank0.jsonl"
    if not os.path.exists(f):
        cand = sorted(glob.glob(f"{d}/{tag}_routing_rank*.jsonl"))
        f = cand[0] if cand else None
    if not f:
        return []
    return [json.loads(l) for l in open(f) if l.strip()]


def fmt_cache(cache):
    return " | ".join(
        f"G{r}[" + ",".join((c if c else "·") for c in cells) + "]"
        for r, cells in enumerate(cache))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--step", type=int, default=None, help="only this fwd step")
    ap.add_argument("--layers", default=None, help="comma list of layer ids")
    ap.add_argument("--max-fetch", type=int, default=12,
                    help="max fetch_ops to print per layer")
    args = ap.parse_args()

    d = args.results_dir
    tag = args.run_tag
    if tag is None:
        cand = glob.glob(f"{d}/*_requests_rank0.json")
        tag = os.path.basename(cand[0]).replace("_requests_rank0.json", "") \
            if cand else "trace"

    reqs = load_requests(d, tag)
    recs = load_global(d, tag)

    print("=" * 78)
    print(f"ROUTING TRACE  tag={tag}  ep_size={recs[0]['ep_size'] if recs else '?'}")
    print("=" * 78)
    print("\n[1] REQUESTS (각 GPU=rank 의 batch)")
    for r in sorted(reqs):
        m = reqs[r]
        print(f"  GPU{r} (rank{r})  batch_size={m['batch_size']}  "
              f"decode_steps={m['decode_steps']}")
        for req in m["requests"]:
            print(f"      {req['request_id']:14s} tok={req['prompt_tokens']:2d}  "
                  f"{req['prompt']!r}")

    layers = None
    if args.layers:
        layers = {int(x) for x in args.layers.split(",")}

    print("\n[2] PER-LAYER ROUTING / CONTROLLER PLAN / ARCHER CACHE")
    print("    (expert 'L:E'  → 실행 GPU.  cache cell = slot 내용 'L:E' 또는 ·)")
    shown = 0
    for rec in recs:
        if args.step is not None and rec["step"] != args.step:
            continue
        if layers is not None and rec["layer"] not in layers:
            continue
        shown += 1
        print("-" * 78)
        print(f"step {rec['step']} ({rec['phase']})  L{rec['layer']}  "
              f"demand={rec['n_demand']} hits={rec['n_hits']} "
              f"misses={rec['n_misses']}")
        # routing summary: GPU → [experts it serves]
        by_gpu = {}
        for e, g in rec["routing"].items():
            by_gpu.setdefault(g, []).append(e)
        for g in sorted(by_gpu):
            es = sorted(by_gpu[g], key=lambda x: int(x.split(":")[1]))
            print(f"   route→GPU{g} ({len(es)}): {','.join(es)}")
        # controller plan
        cp = rec["controller_plan"]
        hits = cp["hit_ops"]
        if hits:
            hs = ", ".join(f"{h['expert']}@G{h['serving_gpu']}s{h['slot']}"
                           for h in hits[:args.max_fetch])
            print(f"   plan.HIT ({len(hits)}): {hs}"
                  + (" …" if len(hits) > args.max_fetch else ""))
        fo = cp["fetch_ops"]
        print(f"   plan.FETCH ({len(fo)}) [order: expert→dst_slot evict victim]:")
        for op in fo[:args.max_fetch]:
            v = op["victim"] if op["victim"] else "—(empty)"
            print(f"       #{op['order']:<3d} {op['expert']:>8s} → "
                  f"G{op['fetcher_gpu']} slot{op['dst_slot']}  evict {v}")
        if len(fo) > args.max_fetch:
            print(f"       … (+{len(fo)-args.max_fetch} more)")
        # archer cache slot updates
        ups = rec["archer_slot_updates"]
        print(f"   archer SLOT updates ({len(ups)}):")
        for u in ups[:args.max_fetch]:
            print(f"       G{u['rank']} slot{u['slot']}: "
                  f"{u['evict'] or '·'} → {u['install'] or '·'}")
        if len(ups) > args.max_fetch:
            print(f"       … (+{len(ups)-args.max_fetch} more)")
        print(f"   archer CACHE after: {fmt_cache(rec['archer_cache_after'])}")
    print("-" * 78)
    print(f"shown {shown} layer-records (of {len(recs)} total). "
          f"raw JSONL: {d}/{tag}_routing_rank*.jsonl")


if __name__ == "__main__":
    main()
