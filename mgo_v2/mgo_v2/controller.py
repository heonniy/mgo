from __future__ import annotations

from typing import Dict

import numpy as np

from .admission import (
    AdmissionContext,
    BalancedRandomAdmission,
    GreedyCurrentAdmission,
    GreedyPathAdmission,
    HungarianAdmission,
    SwapRefinedAdmission,
)
from .affinity import AffinityTables
from .cache import GlobalCacheState
from .config import RuntimeConfig
from .eviction import DiversityEviction, GateHistory, GateScoreEviction, LRUEviction
from .substitution import SubstitutionPolicy, merge_effective_routes
from .types import LayerPlan, LayerRoutes, LocalExecPlan


class GlobalExpertController:
    """The only authority for logical expert residency.

    The controller is deterministic and is intended to be replicated on every
    rank after all ranks have the same global routing metadata.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        similarity: np.ndarray,
        affinity: AffinityTables | None = None,
    ):
        self.config = config
        if similarity.shape != (config.num_layers, config.num_experts, config.num_experts):
            raise ValueError("similarity dimensions do not match the model")
        if not np.isfinite(similarity).all():
            raise ValueError("similarity contains nonfinite values")
        needs_same = config.admission in {"hungarian_same", "hungarian_same_path", "hungarian_swap"}
        needs_path = config.admission in {"greedy_path", "hungarian_same_path", "hungarian_swap"}
        if needs_same and (affinity is None or affinity.same_layer is None):
            raise ValueError(f"{config.admission} requires same_layer affinity")
        if needs_path and (affinity is None or affinity.path is None):
            raise ValueError(f"{config.admission} requires path affinity")
        if affinity is not None:
            for name, table, layers in (("same_layer", affinity.same_layer, config.num_layers),
                                        ("path", affinity.path, config.num_layers - 1)):
                if table is not None and (table.shape != (layers, config.num_experts, config.num_experts)
                                          or not np.isfinite(table).all()):
                    raise ValueError(f"invalid {name} affinity dimensions or nonfinite values")
        self.similarity = similarity
        self.affinity = affinity
        self.cache = GlobalCacheState(config.per_rank_slots())
        self.history = GateHistory(
            config.num_layers, config.num_experts, config.gate_window
        )
        self.substitution = SubstitutionPolicy(
            similarity,
            gate_threshold=config.gate_protect_threshold if config.substitution_enabled else 0.0,
            similarity_threshold=config.similarity_threshold,
        )
        self.eviction = self._build_eviction()
        self.admission = self._build_admission()

    def _build_eviction(self):
        if self.config.eviction == "lru":
            return LRUEviction()
        if self.config.eviction == "gate":
            return GateScoreEviction(self.history)
        if self.config.eviction == "coverage":
            return DiversityEviction(
                self.history,
                self.similarity,
                similarity_threshold=self.config.similarity_threshold,
                lam=self.config.coverage_lambda,
                k_min=self.config.coverage_k,
            )
        raise ValueError(self.config.eviction)

    def _build_admission(self):
        c = self.config
        if c.admission == "random":
            return BalancedRandomAdmission(c.seed)
        if c.admission == "greedy_current":
            return GreedyCurrentAdmission()
        if c.admission == "greedy_path":
            return GreedyPathAdmission(c.path_eta)
        if c.admission == "hungarian_current":
            return HungarianAdmission(name="hungarian_current")
        if c.admission == "hungarian_same":
            return HungarianAdmission(
                use_same=True,
                same_alpha=c.same_layer_alpha,
                name="hungarian_same",
            )
        if c.admission == "hungarian_same_path":
            return HungarianAdmission(
                use_same=True,
                use_path=True,
                same_alpha=c.same_layer_alpha,
                path_eta=c.path_eta,
                name="hungarian_same_path",
            )
        if c.admission == "hungarian_swap":
            seed = HungarianAdmission(
                use_same=True,
                use_path=True,
                same_alpha=c.same_layer_alpha,
                path_eta=c.path_eta,
                name="hungarian_same_path",
            )
            return SwapRefinedAdmission(seed)
        raise ValueError(c.admission)

    def plan_layer(self, routes: LayerRoutes) -> LayerPlan:
        if routes.full_router_probs is not None:
            self.history.update(routes.layer, routes.full_router_probs)

        decision = self.substitution.decide(routes, self.cache)
        effective = merge_effective_routes(routes, decision)
        execution_experts = {e for token in effective for e in token}

        # Owners that already exist before this layer's admissions.
        preowned = {
            e: self.cache.owner_of((routes.layer, e))
            for e in execution_experts
            if self.cache.resident((routes.layer, e))
        }
        preowned = {e: int(r) for e, r in preowned.items() if r is not None}

        incoming = sorted(decision.residual_exact_misses)
        adm_ctx = AdmissionContext(
            layer=routes.layer,
            incoming=incoming,
            origin_ranks=routes.origin_ranks,
            effective_token_routes=effective,
            preowned=preowned,
            cache=self.cache,
            world_size=self.config.world_size,
            affinity=self.affinity,
        )
        admission = self.admission.place(adm_ctx)

        local_exec = {
            r: LocalExecPlan() for r in range(self.config.world_size)
        }
        # Existing executing experts must not be evicted by same-event misses.
        pinned = {(routes.layer, e) for e in execution_experts if e in preowned}

        # Reject an impossible event before modifying logical residency. Hard
        # quotas and pinned execution experts must never cause partial plans.
        for rank, quota in enumerate(admission.quotas):
            available = self.cache.ranks[rank].capacity - sum(
                self.cache.owner_of(key) == rank for key in pinned
            )
            if quota > available:
                raise RuntimeError(
                    f"rank {rank}: admission quota {quota} exceeds {available} unpinned slots; "
                    "increase cache capacity or reduce the event token batch"
                )

        # Existing hits first.
        for e, rank in sorted(preowned.items()):
            slot = self.cache.ranks[rank].slot_of((routes.layer, e))
            local_exec[rank].hit_ops.append((routes.layer, e, slot))

        tick = self.cache.next_tick()
        order_by_rank = [0] * self.config.world_size

        # Admit exact misses in deterministic expert order. Newly admitted
        # execution experts become pinned immediately.
        for e in incoming:
            rank = admission.expert_to_rank[e]
            rank_cache = self.cache.ranks[rank]
            free = rank_cache.free_slot()
            victim_layer = -1
            victim_expert = -1
            if free is None:
                victim = self.eviction.choose(
                    self.cache, rank, routes.layer, pinned
                )
                victim_layer, victim_expert = victim
                _victim_rank, free = self.cache.evict(victim)
                if _victim_rank != rank:
                    raise AssertionError("victim rank mismatch")

            self.cache.place(rank, (routes.layer, e), free, tick)
            pinned.add((routes.layer, e))
            local_exec[rank].miss_ops.append(
                (
                    routes.layer,
                    e,
                    free,
                    victim_layer,
                    victim_expert,
                    order_by_rank[rank],
                )
            )
            order_by_rank[rank] += 1

        # All execution experts now have an owner.
        owners: Dict[int, int] = {}
        for e in execution_experts:
            owner = self.cache.owner_of((routes.layer, e))
            if owner is None:
                raise RuntimeError(f"execution expert {e} has no owner")
            owners[e] = int(owner)
            self.cache.touch((routes.layer, e), tick)

        self.cache.assert_consistent()

        return LayerPlan(
            layer=routes.layer,
            substitution=decision,
            admission=admission,
            owner_by_expert=owners,
            local_exec=local_exec,
            effective_token_routes=effective,
            pinned_keys=pinned,
        )


# Array-backed physical offload controller: sole owner for current/prefetch plans.
# Retains exact baseline admission/eviction implementations for P=0 parity.
from env_offload_policy import Policy
from br_carep_cpu import choose_slot, balanced_assignment
from la_placement import load_assignment
from .predictor import choose_candidates
from .cache import SlotArena

class DecodePrefetchController:
 def __init__(self,capacities,budget,policy,seed,predictor,quota_table=None):
  self.world=len(capacities);self.policy_name=policy;self.seed=seed;self.budget=budget;self.predictor=predictor
  quota_lut=None
  if policy in ('NEAR_PCIE','NEAR_FAST','NEAR_SPLIT'):
   import json
   from pathlib import Path
   if quota_table is None:raise ValueError('quota table required for PCIe quota policies')
   data=json.loads(Path(quota_table).read_text())
   if data.get('status')!='PASS' or data.get('physical_gpus')!=list(range(8)):
    raise ValueError('invalid eight-GPU quota calibration')
   quota_lut=np.asarray(data['quota_lut'][policy],dtype=np.int64)
  self.main=Policy(capacities,np.zeros((48,128,128),np.float32),False,{'BR':0,'CA':1,'LA':4,'OLD_CA':3,'FCA':5,'LA_CA':6,'LA_CA_NEAR':7,'CA_NATIVE':8,'STATIC_MOD':9,'RANDOM_HASH':10,'NEAR_PCIE':11,'NEAR_FAST':12,'MISS_BAL_COMM':13,'BW':14,'MISS_CAP_COMM':15,'HAQ':16,'HAQ_WORST':17,'STATIC_BLOCK':18,'RANDOM':19,'NEAR_SPLIT':20}[policy],seed,quota_lut)
  self.arena=SlotArena(self.main,budget);self.pending=self.arena.reservations;self.counters=dict(issued=0,useful=0,wasted=0,promotions=0,promotion_evictions=0,promotion_victim_reloads=0,mandatory=0,quota_violations=0)
  self.promotion_victims=set();self.event=-1
 def plan_current(self,event,selected,weights,origins,gates):
  self.event=event;layer=event%48;tick=event+1
  active=None
  if self.pending:
   active=np.zeros(128,np.bool_);active[np.unique(selected)]=True
  m=self.main;m.gates[layer]=gates;promotions=[];discard=[]
  for key,(rank,pfslot,target) in sorted(self.pending.items()):
   assert target==layer and event>=48
   if active[key%128]:
    assert m.owner[key]==0
    slot=choose_slot(rank,layer,active,m.slots,m.capacities,m.last,m.gates,True,128);assert slot>=0
    victim=int(m.slots[rank,slot]);row=np.zeros(48,np.float64)
    promotion=self.arena.promote(key,slot,tick)
    promotions.append(promotion);self.counters['useful']+=1;self.counters['promotions']+=1
    if victim>=0:self.promotion_victims.add(victim);self.counters['promotion_evictions']+=1
   else:discard.append(self.arena.discard(key));self.counters['wasted']+=1
  self.pending.clear()
  out=m.apply(event,selected,weights,origins,gates,np.zeros((128,self.world),np.int32));fetches=out[5]
  quotas=np.bincount([f[0] for f in fetches],minlength=self.world)
  if self.policy_name not in ('STATIC_MOD','RANDOM_HASH','HAQ','STATIC_BLOCK','RANDOM','NEAR_SPLIT'):assert quotas.max()-quotas.min()<=1
  if self.policy_name=='NEAR_SPLIT':assert np.array_equal(quotas,self.main.quota_lut[len(fetches)])
  if self.policy_name=='HAQ':assert quotas.max()<=-(-len(fetches)//self.world)
  for rank,key,slot,victim,rep in fetches:
   assert not rep
   if key in self.promotion_victims:self.counters['promotion_victim_reloads']+=1;self.promotion_victims.remove(key)
  self.counters['mandatory']+=len(fetches)
  return out,promotions,discard
 def plan_prefetch_next(self,per_rank_counts):
  layer=self.event%48
  if self.event<48 or layer==47 or self.budget==0:return []
  predicted=self.predictor.predict_next(layer,per_rank_counts)
  candidates=choose_candidates(predicted,self.main.slots.ravel(),self.pending.keys(),layer+1,self.budget,self.world)
  n=len(candidates)
  if not n:return []
  # Fixed-point scores preserve fractional predictor ordering for integer CA/LA.
  demand=np.rint(predicted.T*1_000_000).astype(np.int64)
  if self.policy_name=='BR':
   slots=np.repeat(np.arange(self.world),[n//self.world+(r<n%self.world) for r in range(self.world)]);order=np.random.default_rng(np.random.SeedSequence([self.seed,self.event,991])).permutation(n);assignment=np.empty(n,np.int64);assignment[order]=slots
  elif self.policy_name=='CA_NATIVE':
   from native_ca_assignment import native_balanced_assignment
   assignment=native_balanced_assignment(demand,candidates,self.world,False)
  elif self.policy_name in ('CA','OLD_CA','FCA'):
   # Predictor exposes expert/rank demand but not token co-routing; packet-aware
   # current admission therefore falls back to demand-locality for prefetch.
   assignment=balanced_assignment(demand,candidates,self.world,False)
  else:
   # Load-first policies keep LA prefetch placement; exact token-level communication
   # locality is unavailable to the predictor and prefetch is disabled in strict studies.
   assignment=load_assignment(demand,candidates,self.main.owner,layer+1)
  counts=np.zeros(self.world,np.int64);result=[]
  for expert,rank in zip(candidates,assignment):
   rank=int(rank);key=(layer+1)*128+int(expert);assert self.main.owner[key]==0 and key not in self.pending
   slot=int(counts[rank]);counts[rank]+=1;assert slot<self.budget
   self.arena.reserve(rank,key,slot,layer+1);result.append((rank,key,slot))
  assert counts.max()-counts.min()<=1
  self.counters['issued']+=len(result)
  return result
