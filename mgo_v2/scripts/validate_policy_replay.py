#!/usr/bin/env python3
"""Replay saved simulator states to check substitution and eviction semantics.

Admission order/owners are taken from the reference receipt to isolate A/B.
This is deliberately not a claim of identical random-number generators.
"""
import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np

from mgo_v2.cache import GlobalCacheState
from mgo_v2.eviction import GateHistory, LRUEviction, GateScoreEviction, DiversityEviction
from mgo_v2.substitution import SubstitutionPolicy, merge_effective_routes
from mgo_v2.types import LayerRoutes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--packed", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument("--eviction", choices=["lru", "gate", "coverage"], required=True)
    p.add_argument("--coverage-lambda", type=float, default=2.0)
    p.add_argument("--events", type=int, default=96)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.reference)
    rows = list(csv.DictReader((root / "events.csv").open()))
    cap = json.loads((root / "receipt.json").read_text())["rank_capacities"]
    # Map large diagnostics instead of loading entire experiments into RAM.
    admissions = np.memmap(root / "admission_details.bin", dtype="<f8", mode="r").reshape(-1, 6)
    expected_experts = np.memmap(root / "expert_events.bin", dtype="<f8", mode="r").reshape(-1, 128, 8)
    sim = np.fromfile(args.similarity, dtype="<f4").reshape(48, 128, 128)
    cache = GlobalCacheState(cap)
    history = GateHistory(48, 128)
    substitution = SubstitutionPolicy(sim)
    eviction = {"lru": LRUEviction(), "gate": GateScoreEviction(history),
                "coverage": DiversityEviction(history, sim, lam=args.coverage_lambda)}[args.eviction]
    cursor = serial = evictions = mappings = hits = subhits = misses = 0
    with open(args.packed, "rb") as f:
        ranks, batch, window, total = struct.unpack("<4i", f.read(16))
        for event in range(min(args.events, total)):
            wave, step, layer, n = struct.unpack("<4i", f.read(16))
            means = np.fromfile(f, "<f4", 256).reshape(2, 128)
            origins = np.fromfile(f, "u1", n).astype(np.int64)
            ids = np.fromfile(f, "u1", n * 8).reshape(n, 8).astype(np.int64)
            weights = np.fromfile(f, "<f4", n * 8).reshape(n, 8)
            history.rows[layer].clear()
            history.rows[layer].append(means[0])
            history.sums[layer] = means[0]
            routes = LayerRoutes(layer, origins, ids, weights)
            decision = substitution.decide(routes, cache)
            effective = merge_effective_routes(routes, decision)
            row = rows[event]
            counts = tuple(sum(int(e) in group for e in ids.ravel()) for group in
                           (decision.exact_hits, set(decision.source_to_target), decision.residual_exact_misses))
            assert counts == tuple(int(row[key]) for key in ("hit", "subhit", "miss")), (event, counts)
            hits += len(decision.exact_hits)
            subhits += len(decision.source_to_target)
            misses += len(decision.residual_exact_misses)
            mappings += len(decision.source_to_target)
            post = np.zeros((n, 128), dtype=bool)
            mass = np.zeros(128)
            for token, routed in enumerate(effective):
                for expert, weight in routed.items():
                    post[token, expert] = True
                    mass[expert] += weight
            np.testing.assert_array_equal(post.sum(0), expected_experts[event, :, 5])
            np.testing.assert_allclose(mass, expected_experts[event, :, 6], atol=1e-7, rtol=1e-7)
            active = set(np.flatnonzero(post.any(0)).tolist())
            pinned = {(layer, e) for e in active if cache.resident((layer, e))}
            assigned = {e: cache.owner_of((layer, e)) for _, e in pinned}
            incoming = set(decision.residual_exact_misses)
            for _ in range(int(row["fetch"])):
                record = admissions[cursor]
                cursor += 1
                assert int(record[0]) == event
                uid, rank, expected_victim = map(int, record[3:])
                l, e = divmod(uid, 128)
                assert l == layer and e in incoming
                incoming.remove(e)
                slot = cache.ranks[rank].free_slot()
                victim = None if slot is not None else eviction.choose(cache, rank, layer, pinned)
                assert (-1 if victim is None else victim[0] * 128 + victim[1]) == expected_victim, (event, uid, victim, expected_victim)
                if victim is not None:
                    _, slot = cache.evict(victim)
                    evictions += 1
                cache.place(rank, (layer, e), slot, event)
                serial += 1
                # Reference LRU breaks same-event admission ties by serial.
                cache.ranks[rank].entries[(layer, e)].admitted_at = serial
                pinned.add((layer, e))
                assigned[e] = rank
            assert not incoming
            expected_owners = np.full(128, -1)
            for e, rank in assigned.items():
                expected_owners[e] = rank
                cache.touch((layer, e), event)
            np.testing.assert_array_equal(expected_owners, expected_experts[event, :, 7])
            cache.assert_consistent()
    result = {"status": "PASS", "events": min(args.events, total), "reference": str(root),
              "evictions": evictions, "admissions": cursor, "mappings": mappings,
              "expert_hit": hits, "expert_subhit": subhits, "expert_miss": misses,
              "scope": "recorded admission replay; raw-route A, local-victim B, merged mass and global owners"}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
