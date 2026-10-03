"""Bounded duplicate refresh layered over the frozen d666414 replay."""
from collections import Counter, OrderedDict
import numpy as np
from cache_policy_cpu import CachePolicyReplay
from replica_pareto_cpu import ROW_BYTES, EXPERT_BYTES, traffic
from mgo_v2.types import LayerRoutes
from mgo_v2.substitution import merge_effective_routes


def bits(mask):
    return int.from_bytes(np.packbits(np.asarray(mask,dtype=np.uint8),bitorder='little').tobytes(),'little')


def peer_rows(origins,tokens,dest,occupancy):
    remote=np.arange(occupancy.shape[1])[None,:]!=origins[:,None]
    return int(np.count_nonzero((occupancy>0)&remote)+np.count_nonzero(dest!=origins[tokens]))


class FutureDemand:
    """Only decode routing, with bounded caching of production substitution."""
    def __init__(self, events):
        self.by_layer={l:[e for e in events if e['layer']==l and e['step']>0] for l in range(48)}
        self.effective_cache=OrderedDict()

    def effective(self,replay,item):
        origins=np.asarray(item['origin_ranks'],dtype=np.int64)
        raw=np.asarray(item['raw_selected_experts'],dtype=np.int64)
        if not replay.substitution_enabled:return origins,raw
        layer=item['layer'];mask=bits(replay.copy_count[layer]>0)
        key=(item['event'],mask)
        if key in self.effective_cache:
            value=self.effective_cache.pop(key);self.effective_cache[key]=value;return value
        weights=np.asarray(item['routing_weights'],dtype=np.float32)
        routes=LayerRoutes(layer,origins,raw,weights)
        decision=replay.policy.decide(routes,replay)
        result=(origins,[list(r) for r in merge_effective_routes(routes,decision)])
        self.effective_cache[key]=result
        if len(self.effective_cache)>4096:self.effective_cache.popitem(last=False)
        return result


def project(replay,layer,origins,effective):
    lengths=np.array([len(r) for r in effective],dtype=np.int64)
    tokens=np.repeat(np.arange(len(origins)),lengths)
    selected=np.concatenate(effective).astype(np.int64)
    locations={int(e):np.flatnonzero(selected==e) for e in np.unique(selected)}
    demands=np.bincount(selected*replay.world+origins[tokens],minlength=128*replay.world).reshape(128,replay.world)
    owners=np.zeros((128,replay.world),dtype=bool)
    primary=demands.argmax(axis=1)
    for e in range(128):
        rs=replay.owners.get((layer,e),())
        if rs:owners[e,list(rs)]=True;primary[e]=replay.primary[layer,e]
    dest=np.where(owners[selected,origins[tokens]],origins[tokens],primary[selected])
    occupancy=np.zeros((len(origins),replay.world),dtype=np.int64)
    np.add.at(occupancy,(tokens,dest),1)
    return dict(origins=origins,selected=selected,tokens=tokens,dest=dest,occupancy=occupancy,locations=locations,primary=primary,owners=owners,lengths=lengths)


def removal_rows(p,replay,layer,e,rank):
    """Exact hypothetical removal, including a duplicated primary's promotion."""
    ix=p['locations'].get(e)
    if ix is None:return 0
    surviving=replay.owners[layer,e]-{rank};assert surviving
    primary=replay.primary[layer,e]
    if primary==rank:primary=min(surviving)
    ts=p['tokens'][ix];origin=p['origins'][ts];before=p['dest'][ix]
    after=np.where(np.isin(origin,list(surviving)),origin,primary)
    changed=before!=after;ts=ts[changed];origin=origin[changed];before=before[changed];after=after[changed]
    return int(np.count_nonzero(before!=origin)-np.count_nonzero(after!=origin)
               +np.count_nonzero((before!=origin)&(p['occupancy'][ts,before]==1))
               -np.count_nonzero((after!=origin)&(p['occupancy'][ts,after]==0)))


class DynamicReplay(CachePolicyReplay):
    def __init__(self,capacities,rho,eviction,similarity,substitution,refresh_policy,future=None):
        super().__init__(capacities,rho,eviction,similarity,substitution)
        assert refresh_policy in ('N0','C1','C2','O4','OR')
        assert (future is not None)==(refresh_policy in ('O4','OR'))
        self.refresh_policy=refresh_policy;self.future=future
        self.revisions=[0]*48;self.forecasts={};self.copy_birth={}
        self.refresh_records=[];self.refresh_stats={}

    def place(self,rank,key,choice):
        victim=choice[1]
        if victim is not None:
            self.copy_birth.pop((rank,victim));self.revisions[victim[0]]+=1
        super().place(rank,key,choice)
        self.copy_birth[rank,key]=self.tick-1;self.revisions[key[0]]+=1

    def forecast(self,layer):
        assert self.future is not None
        items=[x for x in self.future.by_layer[layer] if x['event']>self.tick-1]
        if self.refresh_policy=='O4':items=items[:4]
        cache_key=(self.revisions[layer],tuple(x['event'] for x in items))
        if layer in self.forecasts and self.forecasts[layer][0]==cache_key:return self.forecasts[layer][1]
        add=np.zeros((128,self.world),dtype=np.int64);remove={}
        presence={(e,r):0 for e in range(128) for r in range(self.world)}
        isolated=dict(presence);offsets=[0]*self.world
        duplicates=[(e,r) for (l,e),rs in self.owners.items() if l==layer and len(rs)>1 for r in sorted(rs)]
        for item in items:
            origins,effective=self.future.effective(self,item);p=project(self,layer,origins,effective)
            ts=p['tokens'];ds=p['dest'];es=p['selected'];rs=origins[ts]
            remote=ds!=rs
            np.add.at(add,(es[remote],rs[remote]),1+(p['occupancy'][ts[remote],ds[remote]]==1).astype(np.int64))
            for e,r in duplicates:remove[e,r]=remove.get((e,r),0)+removal_rows(p,self,layer,e,r)
            for rank in range(self.world):
                rank_tokens=np.flatnonzero(origins==rank);mapping=np.full(len(origins),-1,dtype=np.int64);mapping[rank_tokens]=np.arange(len(rank_tokens))
                for e,ix in p['locations'].items():
                    ix=ix[rs[ix]==rank]
                    if not len(ix):continue
                    mask=np.zeros(len(rank_tokens),dtype=bool);mask[mapping[ts[ix]]]=True
                    presence[e,rank] |= bits(mask)<<offsets[rank]
                    isolated_ix=ix[(ds[ix]!=rank)&(p['occupancy'][ts[ix],ds[ix]]==1)]
                    mask[:]=False;mask[mapping[ts[isolated_ix]]]=True
                    isolated[e,rank] |= bits(mask)<<offsets[rank]
                offsets[rank]+=len(rank_tokens)
        result=dict(add=add,remove=remove,presence=presence,isolated=isolated,events=[x['event'] for x in items])
        self.forecasts[layer]=(cache_key,result);return result

    def refresh(self,layer,origins,selected,tokens,lengths,dest,occupancy,locations):
        before=peer_rows(origins,tokens,dest,occupancy)*ROW_BYTES
        self.refresh_stats=dict(refresh_admissions=0,refresh_duplicate_evictions=0,refresh_immediate_peer_saved=0,refresh_predicted_horizon_peer_saved=0,pre_refresh_peer_activation_bytes=before)
        if self.refresh_policy=='N0':return [],0
        maximum=2 if self.refresh_policy=='C2' else 1;operations=[];saved=0
        for _ in range(maximum):
            if self.duplicates!=self.cap:break
            victims={r:sorted([k for k in self.ranks[r] if k not in self.active and len(self.owners[k])>1],key=lambda k:(self.ranks[r][k].used,k)) for r in range(self.world)}
            remote=dest!=origins[tokens]
            savings=np.zeros((128,self.world),dtype=np.int64)
            np.add.at(savings,(selected[remote],origins[tokens[remote]]),1+(occupancy[tokens[remote],dest[remote]]==1).astype(np.int64))
            candidates=[(int(e),int(r)) for e,r in zip(*np.nonzero(savings)) if victims[int(r)]]
            if not candidates:break
            chosen=None
            if self.refresh_policy in ('C1','C2'):
                assert self.future is None
                for e,r in candidates:
                    v=victims[r][0];candidate=(-int(savings[e,r]),e,r,self.ranks[r][v].used,v)
                    if chosen is None or candidate<chosen:chosen=candidate
            else:
                forecasts={l:self.forecast(l) for l in {layer}|{v[0] for vs in victims.values() for v in vs}}
                f=forecasts[layer]
                best_other={}
                for r,vs in victims.items():
                    choices=[(-forecasts[v[0]]['remove'].get((v[1],r),0),self.ranks[r][v].used,v) for v in vs if v[0]!=layer]
                    if choices:best_other[r]=min(choices)[2]
                for e,r in candidates:
                    options=[v for v in victims[r] if v[0]==layer]
                    if r in best_other:options.append(best_other[r])
                    for v in options:
                        score=int(savings[e,r])+int(f['add'][e,r])+forecasts[v[0]]['remove'].get((v[1],r),0)
                        if v[0]==layer:
                            new_primary=self.primary[v] if self.primary[v]!=r else min(self.owners[v]-{r})
                            if self.primary[layer,e]==new_primary:score-=(f['isolated'][e,r]&f['presence'][v[1],r]).bit_count()
                        candidate=(-score,e,r,self.ranks[r][v].used,v)
                        if score>0 and (chosen is None or candidate<chosen):chosen=candidate
            if chosen is None:break
            negative,e,r,_,victim=chosen;key=(layer,e);q=self.primary[key]
            ix=locations[e,r];ts=tokens[ix];immediate=int(savings[e,r])*ROW_BYTES
            assert victim not in self.active and len(self.owners[victim])>=2 and r not in self.owners[key]
            unique=set(self.owners);duplicates=self.duplicates;slot=self.ranks[r][victim].slot
            record=dict(event=self.tick-1,step=(self.tick-1)//48,layer=layer,expert=e,rank=r,victim_layer=victim[0],victim_expert=victim[1],victim_copy_age=self.tick-1-self.copy_birth[r,victim],victim_idle_age=self.tick-self.ranks[r][victim].used,victim_primary=self.primary[victim]==r,victim_copy_count=len(self.owners[victim]),immediate_peer_saved=immediate,predicted_horizon_peer_saved=-negative*ROW_BYTES)
            peer_before=peer_rows(origins,tokens,dest,occupancy)*ROW_BYTES
            self.place(r,key,(slot,victim))
            assert set(self.owners)==unique and self.duplicates==duplicates==self.cap
            occupancy[ts,q]-=1;occupancy[ts,r]+=1;dest[ix]=r
            assert peer_before-peer_rows(origins,tokens,dest,occupancy)*ROW_BYTES==immediate
            self.refresh_records.append(record);operations.append(('refresh',e,r,(slot,victim),immediate))
            saved+=immediate
            self.refresh_stats['refresh_admissions']+=1;self.refresh_stats['refresh_duplicate_evictions']+=1
            self.refresh_stats['refresh_immediate_peer_saved']+=immediate;self.refresh_stats['refresh_predicted_horizon_peer_saved']+=-negative*ROW_BYTES
        assert before-peer_rows(origins,tokens,dest,occupancy)*ROW_BYTES==saved
        return operations,saved

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
        added, immediate = self.refresh(layer, origins, selected, tokens, lengths, dest, occupancy, locations)
        fetch['replica_fetches'] += len(added)
        savings_total += immediate
        operations.extend(added)
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
        row.update(self.refresh_stats)
        return row, np.split(dest, np.cumsum(lengths)[:-1]), t, operations
