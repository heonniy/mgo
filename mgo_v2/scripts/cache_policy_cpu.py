"""Generalized historical physical-copy replay; supports merged ragged routes.

No model/CUDA imports. Gate vectors come from exact diagnostic captures;
substitution calls the existing production policy without threshold changes.
"""
from collections import Counter
import numpy as np
from replica_pareto_cpu import ReplicaReplay, ROW_BYTES, EXPERT_BYTES, traffic
from mgo_v2.substitution import SubstitutionPolicy, merge_effective_routes
from mgo_v2.types import LayerRoutes


def midrank2(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (2 * starts + counts - 1)[inverse]


class CachePolicyReplay(ReplicaReplay):
    def __init__(self, capacities, rho, eviction, similarity, substitution=False, fixed_cap=None):
        super().__init__(capacities, rho)
        if fixed_cap is not None:self.cap = fixed_cap
        self.eviction = eviction
        self.similarity = similarity
        self.substitution_enabled = substitution
        self.policy = SubstitutionPolicy(similarity)
        self.gates = np.zeros(similarity.shape[:2], dtype=np.float32)
        self.neighbors = similarity >= np.asarray(.65, dtype=similarity.dtype)
        for layer in self.neighbors:np.fill_diagonal(layer, True)
        self.coverage = np.zeros(similarity.shape[:2], dtype=np.int32)
        self.damage = self.neighbors.sum(axis=1)
        self.copy_count = np.zeros(similarity.shape[:2], dtype=np.int32)
        self.used_array = np.zeros((self.world, *similarity.shape[:2]), dtype=np.int64)
        self.live = {}; self.lives = []; self.evictions = np.zeros(self.world, dtype=np.int64)
        self.event_rows = []; self.similarities = []; self.active = set()
        self.ordered = None

    def resident(self, key):return key in self.owners
    def resident_layer(self, layer):return [e for l,e in self.owners if l == layer]

    def begin_event(self, active):
        self.active = active
        # New admissions are active/pinned: legal victim order for LRU/GATE
        # can only lose its first element within this event.
        self.legal = []
        for rank, entries in enumerate(self.ranks):
            keys = [k for k in entries if k not in active]
            if self.eviction == 'lru':keys.sort(key=lambda k:(entries[k].used,k),reverse=True)
            elif self.eviction == 'gate':keys.sort(key=lambda k:(float(self.gates[k]),entries[k].used,k),reverse=True)
            self.legal.append(keys)
        self.cached_choices = {}

    def slot_or_victim(self, rank, active):
        entries = self.ranks[rank]
        if len(entries) < self.capacities[rank]:return self.slots[rank].index(None), None
        if rank in self.cached_choices:return self.cached_choices[rank]
        legal = self.legal[rank]
        if not legal:return None
        if self.eviction in ('lru','gate'):victim = legal[-1]
        else:
            keys = np.asarray(legal); ls, es = keys.T
            damages = np.where(self.copy_count[ls,es] > 1, 0, self.damage[ls,es])
            scores = midrank2(self.gates[ls,es]) + 2 * midrank2(damages)
            ix = np.lexsort((es,ls,self.used_array[rank,ls,es],scores))[0]
            victim = legal[int(ix)]
        choice = entries[victim].slot, victim
        self.cached_choices[rank] = choice
        return choice

    def place(self, rank, key, choice):
        victim = choice[1]
        assert victim not in self.active
        is_replica = key in self.owners
        changed = set()
        if victim is not None:
            self.evictions[rank] += 1
            self.legal[rank].remove(victim)
            if (rank,victim) in self.live:
                life = self.live.pop((rank,victim));life.update(end=self.tick-1,censored=False)
                self.lives.append(life)
            if self.copy_count[victim] == 1:
                self.coverage[victim[0]] -= self.neighbors[victim[0],:,victim[1]]
                changed.add(victim[0])
            self.copy_count[victim] -= 1
        if not self.copy_count[key]:
            self.coverage[key[0]] += self.neighbors[key[0],:,key[1]]
            changed.add(key[0])
        self.copy_count[key] += 1
        super().place(rank,key,choice)
        self.used_array[rank,key[0],key[1]] = self.tick
        if is_replica:
            self.live[rank,key] = dict(rank=rank,layer=key[0],expert=key[1],birth=self.tick-1,reuse_events=0)
        if self.eviction == 'coverage':
            for layer in changed:
                self.damage[layer] = self.neighbors[layer][self.coverage[layer] <= 1].sum(axis=0)
        if self.eviction == 'coverage':self.cached_choices.clear()
        else:self.cached_choices.pop(rank,None)

    def record_served(self, served):
        for rank,key in served:
            self.used_array[rank,key[0],key[1]] = self.tick
            life = self.live.get((rank,key))
            if life is not None and life['birth'] < self.tick-1:life['reuse_events'] += 1

    def step(self, item):
        layer = item['layer'];self.gates[layer] = item['gate_scores']
        selected = np.asarray(item['raw_selected_experts'],dtype=np.int64)
        weights = np.asarray(item['routing_weights'],dtype=np.float32)
        origins = np.asarray(item['origin_ranks'],dtype=np.int64)
        routes = LayerRoutes(layer, origins, selected, weights)
        # OFF needs no substitution policy call; raw route order is preserved.
        info = dict(substituted_sources=0,substituted_routes=0,substituted_gate_mass=0.,raw_gate_mass=float(weights.sum(dtype=np.float64)),raw_routes=int(selected.size),protected_exact_misses=0,residual_exact_misses=0,tier_active_exact_hit=0,tier_protected_exact_miss=0,tier_inactive_resident=0)
        if self.substitution_enabled:
            decision = self.policy.decide(routes,self)
            merged = merge_effective_routes(routes,decision)
            effective = [list(row) for row in merged]
            info['substituted_sources'] = len(decision.source_to_target)
            info['protected_exact_misses'] = len(decision.protected_misses)
            info['residual_exact_misses'] = len(decision.residual_exact_misses)
            for source,target in decision.source_to_target.items():
                mask = selected == source
                assert float(weights[mask].max()) < .20
                score = float(self.similarity[layer,source,target]);assert score >= self.policy.similarity_threshold
                info['substituted_routes'] += int(mask.sum())
                info['substituted_gate_mass'] += float(weights[mask].sum(dtype=np.float64))
                info['tier_'+decision.target_tier[source]] += 1
                self.similarities.append((self.tick,score))
        else:effective = selected
        before = self.evictions.copy()
        row, dest, traffic_result, ops = self.event(layer,origins,effective)
        row.update(info,event=self.tick-1,step=(self.tick-1)//48,rank_evictions=(self.evictions-before).tolist())
        self.event_rows.append(row)
        return row, dest, traffic_result, ops

    def lifecycle(self):
        return self.lives + [dict(life,end=self.tick,censored=True) for life in self.live.values()]

    def event(self, layer, origins, selected, audit_greedy=False, candidate_score=None, owner_selector=None):
        origins = np.asarray(origins, dtype=np.int64)
        assert len(selected) == len(origins)
        lengths = np.array([len(row) for row in selected], dtype=np.int64)
        assert all(len(set(map(int, row))) == len(row) for row in selected)
        tokens = np.repeat(np.arange(len(origins)), lengths)
        selected = np.concatenate(selected).astype(np.int64)
        assert np.all((origins >= 0) & (origins < self.world))
        experts = sorted(map(int, np.unique(selected)))
        active = {(layer, e) for e in experts}
        counts = {e: np.bincount(origins[tokens[selected == e]], minlength=self.world)
                  for e in experts}
        locations = {(e, r): np.flatnonzero((selected == e) & (origins[tokens] == r))
                     for e in experts for r in range(self.world) if counts[e][r]}
        # Freeze pre-event residency for hit accounting. Active copies are pinned.
        pre = {key: set(self.owners.get(key, ())) for key in active}
        first = {e: int(np.argmax(counts[e])) for e in experts if not pre[layer, e]}
        if owner_selector is not None:
            chosen = owner_selector(self, layer, origins, selected, counts, dict(first))
            assert set(chosen) == set(first)
            assert all(0 <= r < self.world and counts[e][r] > 0 for e, r in chosen.items())
            first = chosen
        # Atomic failure: preflight all mandatory destinations before ANY mutation.
        needed = Counter(first.values())
        for r, n in needed.items():
            available = self.capacities[r] - sum(k in active for k in self.ranks[r])
            if n > available:
                raise RuntimeError(f'no legal capacity on rank {r}: {n} mandatory copies, {available} slots')
        self.begin_event(active)
        self.tick += 1
        fetch = dict(first_copy_fetches=0, reload_fetches=0, replica_fetches=0)
        operations = []
        duplicate_peak = self.duplicates
        for e, rank in sorted(first.items()):
            key = layer, e
            category = 'reload_fetches' if key in self.seen else 'first_copy_fetches'
            choice = self.slot_or_victim(rank, active)
            assert choice is not None
            self.place(rank, key, choice)
            fetch[category] += 1
            operations.append(('first', e, rank, choice))
        dest = np.empty_like(selected)
        for (e, rank), ix in locations.items():
            dest[ix] = rank if rank in self.owners[layer, e] else self.primary[layer, e]
        occupancy = np.zeros((len(origins), self.world), dtype=np.int64)
        np.add.at(occupancy, (tokens, dest), 1)
        # A local replica moves only origin-rank r's routes from primary to r.
        # Each moved expert row saves one combine row. A dispatch row is saved
        # precisely when no other expert for that token remains on the old rank.
        savings_total = 0
        route_candidate = selected * self.world + origins[tokens]
        while self.duplicates < self.cap:
            legal_ranks = np.array([len(self.ranks[r]) < self.capacities[r] or bool(self.legal[r]) for r in range(self.world)])
            remote = (dest != origins[tokens]) & legal_ranks[origins[tokens]]
            if not remote.any():break
            weights = 1 + (occupancy[tokens[remote],dest[remote]] == 1).astype(np.int64)
            # Integer add.at avoids float price/ranking arithmetic. Flattened
            # expert*world+rank gives the historical expert-then-rank tie-break.
            savings = np.zeros(self.similarity.shape[1] * self.world,dtype=np.int64)
            np.add.at(savings,route_candidate[remote],weights)
            winner = int(np.argmax(savings))
            e,r = divmod(winner,self.world)
            key = layer,e; q = self.primary[key]
            ix = locations[e,r];ts = tokens[ix]
            choice = self.slot_or_victim(r,active);assert choice is not None
            assert candidate_score is None, 'This frozen study uses historical current-byte greedy only'
            if audit_greedy:
                trial=dest.copy();trial[ix]=r
                before=traffic(origins,np.split(dest,np.cumsum(lengths)[:-1]),self.world)['peer_bytes']
                after=traffic(origins,np.split(trial,np.cumsum(lengths)[:-1]),self.world)['peer_bytes']
                assert before-after == int(savings[winner])*ROW_BYTES
            # The selection score may be future-aware; traffic accounting always
            # uses the exact current-event marginal, independent of that score.
            actual_saving = (len(ts) + int((occupancy[ts, q] == 1).sum())) * ROW_BYTES
            self.place(r, key, choice)
            assert self.duplicates <= self.cap
            duplicate_peak = max(duplicate_peak, self.duplicates)
            occupancy[ts, q] -= 1; occupancy[ts, r] += 1
            dest[ix] = r
            fetch['replica_fetches'] += 1
            savings_total += actual_saving
            operations.append(('replica', e, r, choice, actual_saving))
        assert np.all(occupancy >= 0)
        # LRU means physical usage: touch only the copies that actually serve.
        local_hits = remote_resident = pre_global = local_service = 0
        served = set()
        for origin, e, dst in zip(origins[tokens], selected, dest):
            key = layer, int(e);dst = int(dst);origin = int(origin)
            assert dst in self.owners[key]
            assert dst == (origin if origin in self.owners[key] else self.primary[key])
            local_hits += origin in pre[key]
            pre_global += bool(pre[key])
            remote_resident += dst != origin and dst in pre[key]
            local_service += dst == origin
            served.add((dst, key))
        for rank, key in served:
            self.ranks[rank][key].used = self.tick
        self.record_served(served)
        self.validate()
        t = traffic(origins, np.split(dest, np.cumsum(lengths)[:-1]), self.world)
        row = dict(**fetch, total_fetches=sum(fetch.values()),
                   expert_h2d_bytes=sum(fetch.values()) * EXPERT_BYTES,
                   peer_activation_bytes=t['peer_bytes'], dispatch_bytes=t['dispatch_bytes'],
                   combine_bytes=t['combine_bytes'], remote_token_rank_pairs=t['remote_pairs'],
                   remote_expert_routes=t['remote_expert_routes'], raw_expert_routes=int(selected.size),
                   pre_event_local_exact_hits=local_hits, pre_event_global_hits=pre_global,
                   global_resident_remote_services=remote_resident, final_local_services=local_service,
                   duplicate_slots=self.duplicates, duplicate_peak_in_event=max(duplicate_peak,self.duplicates),
                   resident_copies=sum(map(len,self.ranks)), unique_resident_experts=len(self.owners),
                   greedy_peer_bytes_saved=savings_total, global_miss_expert_events=len(first))
        return row, np.split(dest, np.cumsum(lengths)[:-1]), t, operations
