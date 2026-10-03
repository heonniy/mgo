"""Single-copy owner scoring; all physical mutations remain in ReplicaReplay."""
import bisect
import numpy as np
from replica_pareto_cpu import ReplicaReplay, ROW_BYTES, traffic

POLICIES = {'F': 0, 'O0': 0, 'OH1': 1, 'OH2': 2, 'OH4': 4, 'OHremaining': 8}


def owner_costs(origins, selected, destinations, expert, world=4):
    """Candidate-dependent exact wire cost, excluding constant other routes."""
    ts, ks = np.where(selected == expert)
    if not len(ts):
        return np.zeros(world, dtype=np.int64)
    others = destinations[ts].copy()
    others[np.arange(len(ts)), ks] = -1
    return np.array([sum((origins[ts] != rank) *
                        (1 + ~np.any(others == rank, axis=1))) * ROW_BYTES
                     for rank in range(world)], dtype=np.int64)


class PlacementReplay(ReplicaReplay):
    def __init__(self, capacities, policy, future, demand):
        super().__init__(capacities, 0)
        self.policy = policy
        self.future = future
        self.demand = demand
        self.index = -1
        self.decisions = []
        self.live = {}
        self.lifetimes = []
        self.evictions = [0] * self.world
        self.event_decisions = []

    def choose(self, replay, layer, origins, selected, counts, first):
        owners = {e: self.primary.get((layer, e), first.get(e)) for e in counts}
        dest = np.array([[owners[int(e)] for e in row] for row in selected])
        for e in sorted(first):
            baseline = first[e]
            costs = owner_costs(origins, selected, dest, e, self.world)
            if self.policy != 'O0':
                costs += self.future[(self.index, e, POLICIES[self.policy])]
            candidates = np.flatnonzero(counts[e])
            rank = min(map(int, candidates), key=lambda r: (int(costs[r]), -int(counts[e][r]), r))
            first[e] = rank
            dest[selected == e] = rank
            self.event_decisions.append(dict(event=self.index, layer=layer, expert=e,
                owner=rank, F_owner=baseline, changed=rank != baseline,
                predicted_costs=costs.tolist(), current_counts=counts[e].tolist()))
        return first

    def place(self, rank, key, choice):
        assert key not in self.owners, 'admission without global miss / duplicate'
        victim = choice[1]
        if victim is not None:
            assert victim not in self.active, 'active expert eviction'
            if self.index >= 48:
                self.evictions[rank] += 1
            if victim in self.live:
                record = self.live.pop(victim)
                record['end_event'] = self.index
                record['right_censored'] = False
                self.lifetimes.append(record)
        super().place(rank, key, choice)
        assert len(self.owners[key]) == 1

    def step(self, index, layer, origins, selected):
        self.index = index
        self.active = {(layer, int(e)) for e in np.unique(selected)}
        before = {key: next(iter(owners)) for key, owners in self.owners.items()}
        for key in self.active:
            if key in self.live and self.live[key]['next_demand_event'] == index:
                self.live[key]['survived_next_demand'] = True
        self.event_decisions = []
        callback = self.choose if index >= 48 and self.policy != 'F' else None
        row, dest, tr, ops = super().event(layer, origins, selected, owner_selector=callback)
        if index >= 48:
            if self.policy == 'F':
                self.event_decisions = [dict(event=index, layer=layer, expert=e,
                    owner=r, F_owner=r, changed=False) for _, e, r, *_ in ops]
            for decision in self.event_decisions:
                key = layer, decision['expert']
                upcoming = self.demand[key]
                pos = bisect.bisect_right(upcoming, index)
                record = dict(decision, birth_event=index,
                    next_demand_event=upcoming[pos] if pos < len(upcoming) else None,
                    survived_next_demand=False)
                self.live[key] = record
            self.decisions.extend(self.event_decisions)
        for key, owner in before.items():
            if key in self.owners:
                assert self.owners[key] == {owner}, 'migration'
        assert not self.duplicates and all(len(rs) == 1 for rs in self.owners.values())
        counter = dest.copy()
        for key in self.active:
            record = self.live.get(key)
            if record and record['changed']:
                counter[selected == key[1]] = record['F_owner']
        row['owner_attributable_peer_bytes_saved'] = traffic(origins, counter, self.world)['peer_bytes'] - tr['peer_bytes'] if index >= 48 else 0
        row['owner_changes'] = sum(d['changed'] for d in self.event_decisions)
        row['rank_unique'] = list(map(len, self.ranks))
        return row, dest, tr, ops

    def finish(self):
        return self.lifetimes + [dict(v, end_event=432, right_censored=True) for v in self.live.values()]
