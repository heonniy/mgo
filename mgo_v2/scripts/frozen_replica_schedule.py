"""Experiment-only frozen action codec/applier. Standard library only; no search."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


class FrozenReplicaState:
    def __init__(self, capacities, cap):
        self.slots = [[None] * n for n in capacities]
        self.entries = [{} for _ in capacities]
        self.owners = {}
        self.primary = {}
        self.seen = set()
        self.tick = 0
        self.cap = cap

    def state(self):
        return (self.tick, [sorted((k, s, u) for k, (s, u) in es.items()) for es in self.entries],
                sorted(self.primary.items()), sorted(self.seen))

    def apply(self, event):
        assert event['event'] == self.tick
        self.tick += 1
        layer = event['layer']
        classes = [dict(first_copy_fetches=0, reload_fetches=0, replica_fetches=0) for _ in self.slots]
        active = {(layer, e) for token in event['selected'] for e in token}
        for op in event['admissions']:
            r, e, slot = op['rank'], op['expert'], op['slot']
            key = layer, e
            victim = tuple(op['victim']) if op['victim'] is not None else None
            assert self.slots[r][slot] == victim and victim not in active
            assert key not in self.entries[r]
            category = ('replica_fetches' if key in self.owners else
                        'reload_fetches' if key in self.seen else 'first_copy_fetches')
            assert category == op['category']
            if victim is not None:
                self.owners[victim].remove(r); del self.entries[r][victim]
                if not self.owners[victim]:
                    del self.owners[victim]; del self.primary[victim]
                elif self.primary[victim] == r:
                    self.primary[victim] = min(self.owners[victim])
            if key not in self.owners:
                self.owners[key] = set(); self.primary[key] = r
            self.owners[key].add(r); self.seen.add(key)
            self.slots[r][slot] = key; self.entries[r][key] = (slot, self.tick)
            classes[r][category] += 1
            assert sum(map(len, self.entries)) - len(self.owners) <= self.cap
        served = set()
        for origin, es, ds in zip(event['origins'], event['selected'], event['destinations']):
            assert len(es) == len(ds) and len(set(es)) == len(es)
            for e, dst in zip(es, ds):
                key = layer, e
                assert dst == (origin if origin in self.owners[key] else self.primary[key])
                slot, _ = self.entries[dst][key]
                self.entries[dst][key] = slot, self.tick
                served.add((dst, e))
        for r, plan in enumerate(event['local_exec']):
            miss = {(op[0], op[1], op[2]) for op in plan['miss_ops']}
            hits = {tuple(op) for op in plan['hit_ops']}
            expected = {(layer, e, self.entries[r][layer, e][0]) for rank, e in served if rank == r}
            assert not hits & miss and hits | miss == expected
            admissions = [op for op in event['admissions'] if op['rank'] == r]
            assert plan['miss_ops'] == [[layer, op['expert'], op['slot'], *(op['victim'] or [-1, -1]), i]
                                        for i, op in enumerate(admissions)]
            assert sorted([*key, slot] for key, (slot, _) in self.entries[r].items()) == event['rank_cache'][r]
            assert classes[r] == event['rank_fetch_classes'][r]
        assert digest(self.state()) == event['post_state_sha256']
        return classes


def compile_event(i, layer, origins, selected, replay, row, dest, traffic, operations, seen_before):
    admissions = []; local = [dict(hit_ops=[], miss_ops=[]) for _ in replay.ranks]
    for kind, e, r, choice, *rest in operations:
        slot, victim = choice
        category = 'replica_fetches' if kind == 'replica' else 'reload_fetches' if (layer, e) in seen_before else 'first_copy_fetches'
        op = dict(rank=r, expert=e, slot=slot, victim=list(victim) if victim is not None else None, category=category)
        admissions.append(op)
        local[r]['miss_ops'].append([layer, e, slot, *(list(victim) if victim is not None else [-1, -1]), len(local[r]['miss_ops'])])
    served = sorted({(int(dst), int(e)) for es, ds in zip(selected, dest) for e, dst in zip(es, ds)})
    misses = {(a['rank'], a['expert']) for a in admissions}
    assert misses <= set(served)
    for r, e in served:
        if (r, e) not in misses:
            local[r]['hit_ops'].append([layer, e, replay.ranks[r][layer, e].slot])
    local_owner = [{} for _ in replay.ranks]
    for origin, es, ds in zip(origins, selected, dest):
        for e, d in zip(es, ds):
            e, d, origin = int(e), int(d), int(origin)
            assert local_owner[origin].get(str(e), d) == d
            local_owner[origin][str(e)] = d
    classes = [dict(first_copy_fetches=0, reload_fetches=0, replica_fetches=0) for _ in replay.ranks]
    for a in admissions: classes[a['rank']][a['category']] += 1
    event = dict(event=i, layer=layer, step=i // 48, origins=origins.tolist(), selected=selected.tolist(),
                 destinations=dest.tolist(), admissions=admissions, local_exec=local,
                 local_owner=local_owner, dispatch_counts=traffic['dispatch'].tolist(),
                 combine_counts=traffic['combine'].tolist(), rank_fetch_classes=classes,
                 rank_cache=[sorted([*k, e.slot] for k, e in entries.items()) for entries in replay.ranks],
                 post_state_sha256=digest(replay.state()), counters=row)
    return event
