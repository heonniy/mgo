#!/usr/bin/env python3
"""CPU-only trace screen; no runtime, model, Torch or CUDA imports."""
from collections import Counter
from dataclasses import dataclass
import math
import numpy as np

ROW_BYTES = 2048 * 2
EXPERT_BYTES = 9 * 1024**2


def traffic(origins, destinations, world):
    """Activation-only sends, plus independently counted receives."""
    dispatch = np.zeros((world, world), dtype=np.int64)
    combine = np.zeros_like(dispatch)
    dispatch_recv = np.zeros_like(dispatch)
    combine_recv = np.zeros_like(dispatch)
    for origin, ds in zip(origins, destinations):
        origin = int(origin)
        for dst in set(map(int, ds)):
            dispatch[origin, dst] += 1
            dispatch_recv[dst, origin] += 1
        for dst in ds:
            dst = int(dst)
            combine[dst, origin] += 1
            combine_recv[origin, dst] += 1
    assert np.array_equal(dispatch, dispatch_recv.T)
    assert np.array_equal(combine, combine_recv.T)
    remote = ~np.eye(world, dtype=bool)
    pairs = int(dispatch[remote].sum())
    returns = int(combine[remote].sum())
    return dict(dispatch=dispatch, combine=combine,
                dispatch_bytes=pairs * ROW_BYTES, combine_bytes=returns * ROW_BYTES,
                peer_bytes=(pairs + returns) * ROW_BYTES, remote_pairs=pairs,
                remote_expert_routes=returns)


@dataclass
class Entry:
    slot: int
    used: int


class ReplicaReplay:
    def __init__(self, capacities, rho):
        if not 0 <= rho <= 1:
            raise ValueError('invalid replica fraction')
        self.capacities = list(capacities)
        self.world = len(capacities)
        self.capacity = sum(capacities)
        self.cap = math.floor(rho * self.capacity)
        self.ranks = [{} for _ in capacities]
        self.slots = [[None] * n for n in capacities]
        self.owners = {}
        self.primary = {}
        self.seen = set()
        self.tick = 0
        self.duplicates = 0

    def slot_or_victim(self, rank, active):
        slots, entries = self.slots[rank], self.ranks[rank]
        if len(entries) < len(slots):
            return slots.index(None), None
        legal = [k for k in entries if k not in active]
        if not legal:
            return None
        victim = min(legal, key=lambda k: (entries[k].used, k))
        return entries[victim].slot, victim

    def place(self, rank, key, choice):
        """Apply a prevalidated choice; never evict an active key."""
        assert rank not in self.owners.get(key, ())
        slot, victim = choice
        if victim is not None:
            old = self.owners[victim]
            self.duplicates -= len(old) > 1
            old.remove(rank)
            del self.ranks[rank][victim]
            if old:
                if self.primary[victim] == rank:
                    self.primary[victim] = min(old)
            else:
                del self.owners[victim]
                del self.primary[victim]
        existing = self.owners.get(key)
        if existing:
            existing.add(rank)
            self.duplicates += 1
        else:
            self.owners[key] = {rank}
            self.primary[key] = rank
        self.slots[rank][slot] = key
        self.ranks[rank][key] = Entry(slot, self.tick)
        self.seen.add(key)

    def validate(self):
        owners = {}
        for rank, entries in enumerate(self.ranks):
            assert len(entries) <= self.capacities[rank]
            assert sum(k is not None for k in self.slots[rank]) == len(entries)
            for key, e in entries.items():
                assert self.slots[rank][e.slot] == key
                owners.setdefault(key, set()).add(rank)
        assert owners == self.owners
        assert set(self.primary) == set(owners)
        assert all(self.primary[k] in rs for k, rs in owners.items())
        duplicates = sum(map(len, self.ranks)) - len(owners)
        assert self.duplicates == duplicates and 0 <= duplicates <= self.cap

    def state(self):
        return (self.tick, [[(k, e.slot, e.used) for k, e in sorted(rank.items())] for rank in self.ranks],
                sorted(self.primary.items()), sorted(self.seen))

    def event(self, layer, origins, selected, audit_greedy=False, candidate_score=None):
        origins = np.asarray(origins, dtype=np.int64)
        selected = np.asarray(selected, dtype=np.int64)
        assert selected.ndim == 2 and len(selected) == len(origins)
        assert np.all((origins >= 0) & (origins < self.world))
        assert all(len(set(map(int, row))) == len(row) for row in selected)
        experts = sorted(map(int, np.unique(selected)))
        active = {(layer, e) for e in experts}
        counts = {e: np.bincount(origins[np.any(selected == e, axis=1)], minlength=self.world)
                  for e in experts}
        locations = {(e, r): np.where((selected == e) & (origins[:, None] == r))
                     for e in experts for r in range(self.world) if counts[e][r]}
        # Freeze pre-event residency for hit accounting. Active copies are pinned.
        pre = {key: set(self.owners.get(key, ())) for key in active}
        first = {e: int(np.argmax(counts[e])) for e in experts if not pre[layer, e]}
        # Atomic failure: preflight all mandatory destinations before ANY mutation.
        needed = Counter(first.values())
        for r, n in needed.items():
            available = self.capacities[r] - sum(k in active for k in self.ranks[r])
            if n > available:
                raise RuntimeError(f'no legal capacity on rank {r}: {n} mandatory copies, {available} slots')
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
        for (e, rank), (ts, ks) in locations.items():
            dest[ts, ks] = rank if rank in self.owners[layer, e] else self.primary[layer, e]
        occupancy = np.zeros((len(origins), self.world), dtype=np.int64)
        np.add.at(occupancy, (np.repeat(np.arange(len(origins)), selected.shape[1]), dest.ravel()), 1)
        # A local replica moves only origin-rank r's routes from primary to r.
        # Each moved expert row saves one combine row. A dispatch row is saved
        # precisely when no other expert for that token remains on the old rank.
        savings_total = 0
        while self.duplicates < self.cap:
            choices = {r: self.slot_or_victim(r, active) for r in range(self.world)}
            best = None
            for e in experts:
                key = layer, e
                q = self.primary[key]
                for r in range(self.world):
                    if not counts[e][r] or r in self.owners[key] or choices[r] is None:
                        continue
                    ts, ks = locations[e, r]
                    assert np.all(dest[ts, ks] == q) and q != r
                    saving = (len(ts) + int((occupancy[ts, q] == 1).sum())) * ROW_BYTES
                    if audit_greedy:
                        trial = dest.copy(); trial[ts, ks] = r
                        assert traffic(origins, dest, self.world)['peer_bytes'] - traffic(origins, trial, self.world)['peer_bytes'] == saving
                    score = saving if candidate_score is None else candidate_score(self, layer, e, r, choices[r], saving)
                    candidate = (-score, e, r)
                    if score > 0 and (best is None or candidate < best):
                        best = candidate
            if best is None:
                break
            negative, e, r = best
            key = layer, e
            q = self.primary[key]
            ts, ks = locations[e, r]
            choice = choices[r]
            # The selection score may be future-aware; traffic accounting always
            # uses the exact current-event marginal, independent of that score.
            actual_saving = (len(ts) + int((occupancy[ts, q] == 1).sum())) * ROW_BYTES
            self.place(r, key, choice)
            assert self.duplicates <= self.cap
            duplicate_peak = max(duplicate_peak, self.duplicates)
            occupancy[ts, q] -= 1; occupancy[ts, r] += 1
            dest[ts, ks] = r
            fetch['replica_fetches'] += 1
            savings_total += actual_saving
            operations.append(('replica', e, r, choice, actual_saving))
        assert np.all(occupancy >= 0)
        # LRU means physical usage: touch only the copies that actually serve.
        local_hits = remote_resident = pre_global = local_service = 0
        served = set()
        for origin, es, ds in zip(origins, selected, dest):
            for e, dst in zip(es, ds):
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
        self.validate()
        t = traffic(origins, dest, self.world)
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
        return row, dest, t, operations


class IndependentZeroReplay:
    """Separately implemented rho=0 reference: slot arrays and Python counters."""
    def __init__(self, capacities):
        self.slots = [[None]*n for n in capacities]
        self.used = [[0]*n for n in capacities]
        self.seen = set()
        self.tick = 0

    def event(self, layer, origins, selected):
        demand = {}
        for origin, experts in zip(origins, selected):
            for e in experts:
                demand.setdefault(int(e), Counter())[int(origin)] += 1
        owners = {key: (r, slot) for r, slots in enumerate(self.slots)
                  for slot,key in enumerate(slots) if key is not None}
        active = {(layer,e) for e in demand}
        adds = []
        for e in sorted(demand):
            if (layer,e) not in owners:
                rank = min(range(len(self.slots)),key=lambda r:(-demand[e][r],r))
                adds.append((e,rank))
        for r in range(len(self.slots)):
            assert sum(rank==r for _,rank in adds) <= sum(k not in active for k in self.slots[r])
        self.tick += 1
        first=reloads=0
        for e,r in adds:
            key=(layer,e)
            legal=[i for i,k in enumerate(self.slots[r]) if k not in active]
            slot=min(legal,key=lambda i:(0,i) if self.slots[r][i] is None else (1,self.used[r][i],self.slots[r][i]))
            old=self.slots[r][slot]
            if old is not None:del owners[old]
            self.slots[r][slot]=key;self.used[r][slot]=self.tick;owners[key]=(r,slot)
            if key in self.seen:reloads+=1
            else:first+=1
            self.seen.add(key)
        ds=[];dispatch=[[0]*len(self.slots) for _ in self.slots];combine=[[0]*len(self.slots) for _ in self.slots]
        for origin, experts in zip(origins,selected):
            destinations=[]
            for e in experts:
                r,s=owners[layer,int(e)];self.used[r][s]=self.tick
                destinations.append(r);combine[r][int(origin)]+=1
            for r in set(destinations):dispatch[int(origin)][r]+=1
            ds.append(destinations)
        state=(self.tick,[sorted((k,i,self.used[r][i]) for i,k in enumerate(slots) if k is not None) for r,slots in enumerate(self.slots)],
               sorted((k,r) for k,(r,_) in owners.items()), sorted(self.seen))
        return state, np.array(ds),np.array(dispatch),np.array(combine),first,reloads
