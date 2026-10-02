"""Compact integer controller decisions, independent of any transport library."""
from __future__ import annotations
import numpy as np
from .admission import AdmissionContext, balanced_quotas
from .rank_oracle import FrozenRankDemandOracle
from .substitution import merge_effective_routes
from .types import AdmissionResult, LayerPlan, LocalExecPlan, SubstitutionResult

MAGIC = 0x4D474F
VERSION = 1
FIELDS = 9
POLICIES = ('random', 'hungarian_current', 'exact_rank_demand_oracle')
TIERS = ('active_exact_hit', 'protected_exact_miss', 'inactive_resident')


def payload_size(config):
    return 8 + config.world_size + FIELDS * config.num_experts


def encode_plan(plan, config, tick):
    data = np.full(payload_size(config), -1, dtype=np.int32)
    header = data[:8 + config.world_size]
    header[:8] = [MAGIC, VERSION, 0, plan.layer, tick,
                  POLICIES.index(plan.admission.policy_name), config.world_size, config.num_experts]
    header[8:] = plan.admission.quotas
    rows = data[len(header):].reshape(config.num_experts, FIELDS)
    rows[:, 2] = 0
    for source, target in plan.substitution.source_to_target.items():
        rows[source, :2] = target, TIERS.index(plan.substitution.target_tier[source])
    for bit, values in enumerate((plan.substitution.exact_hits, plan.substitution.protected_misses,
                                  plan.substitution.low_misses, plan.substitution.residual_exact_misses)):
        for expert in values:
            rows[expert, 2] |= 1 << bit
    for expert, rank in plan.owner_by_expert.items():
        rows[expert, 3] = rank
    for expert, rank in plan.admission.expert_to_rank.items():
        rows[expert, 4] = rank
    for rank, local in plan.local_exec.items():
        for layer, expert, slot in local.hit_ops:
            assert layer == plan.layer and rows[expert, 3] == rank
            rows[expert, 5] = slot
        for layer, expert, slot, vl, ve, order in local.miss_ops:
            assert layer == plan.layer and rows[expert, 3] == rank
            rows[expert, 5:] = slot, vl, ve, order
    return data


def decode_apply(data, controller, routes):
    """Validate a complete decision before applying its cache mutations.

    Followers reconstruct effective weights with the original summation order.
    They do not run substitution selection, admission optimization or eviction.
    """
    c, cache = controller.config, controller.cache
    if data.dtype != np.int32 or data.shape != (payload_size(c),):
        raise ValueError('invalid plan payload shape/dtype')
    header = data[:8 + c.world_size]
    magic, version, status, layer, tick, policy, world, experts = map(int, header[:8])
    if status != 0:
        raise RuntimeError('planner rank failed before broadcasting a decision')
    if (magic, version, layer, tick, world, experts) != (MAGIC, VERSION, routes.layer, cache.tick + 1, c.world_size, c.num_experts):
        raise ValueError('plan header/state mismatch')
    if policy < 0 or policy >= len(POLICIES) or POLICIES[policy] != controller.admission.name:
        raise ValueError('plan policy mismatch')
    rows = data[len(header):].reshape(experts, FIELDS)
    mapping, tiers = {}, {}
    sets = [set() for _ in range(4)]
    owners, assignment = {}, {}
    for expert, row in enumerate(rows):
        target, tier, flags, owner, incoming, slot, vl, ve, order = map(int, row)
        if not 0 <= flags < 16:
            raise ValueError('invalid substitution flags')
        for bit in range(4):
            if flags & (1 << bit): sets[bit].add(expert)
        if target != -1:
            if not 0 <= target < experts or target == expert or not 0 <= tier < len(TIERS):
                raise ValueError('invalid substitution target/tier')
            mapping[expert], tiers[expert] = target, TIERS[tier]
        elif tier != -1:
            raise ValueError('tier without substitution target')
        if owner != -1:
            if not 0 <= owner < world or not 0 <= slot < cache.ranks[owner].capacity:
                raise ValueError('invalid execution owner/slot')
            owners[expert] = owner
        elif any(x != -1 for x in (incoming, slot, vl, ve, order)):
            raise ValueError('operations without execution owner')
        if incoming != -1:
            if incoming != owner:
                raise ValueError('admission/execution owner mismatch')
            assignment[expert] = incoming
        elif vl != -1 or ve != -1 or order != -1:
            raise ValueError('eviction/order on a hit')
    exact, protected, low, residual = sets
    active = set(routes.selected_experts.ravel().tolist())
    if exact | protected | low != active or exact & (protected | low) or protected & low:
        raise ValueError('invalid substitution partition')
    if not set(mapping) <= low or residual != (protected | (low - set(mapping))):
        raise ValueError('invalid residual/substitution partition')
    if exact != {e for e in active if cache.resident((layer, e))}:
        raise ValueError('exact-hit residency mismatch')
    if set(assignment) != residual:
        raise ValueError('admission/residual mismatch')
    substitution = SubstitutionResult(mapping, tiers, exact, protected, low, residual)
    effective = merge_effective_routes(routes, substitution)
    if set(owners) != {e for token in effective for e in token}:
        raise ValueError('execution demand mismatch')
    quotas = list(map(int, header[8:]))
    if quotas != balanced_quotas(len(assignment), world) or [list(assignment.values()).count(r) for r in range(world)] != quotas:
        raise ValueError('quota mismatch')
    local = {r: LocalExecPlan() for r in range(world)}
    preowned = {}
    occupied = [set() for _ in range(world)]
    orders = [0] * world
    pinned = {(layer, e) for e in owners}
    for expert in sorted(owners):
        rank = owners[expert]
        slot, vl, ve, order = map(int, rows[expert, 5:])
        if slot in occupied[rank]:
            raise ValueError('duplicate execution slot')
        occupied[rank].add(slot)
        key = (layer, expert)
        if expert not in assignment:
            if cache.owner_of(key) != rank or cache.ranks[rank].slot_of(key) != slot:
                raise ValueError('existing owner/slot mismatch')
            preowned[expert] = rank
            local[rank].hit_ops.append((layer, expert, slot))
        else:
            if cache.resident(key) or order != orders[rank]:
                raise ValueError('invalid admission/order')
            orders[rank] += 1
            victim = cache.ranks[rank].slots[slot]
            if victim is None:
                if (vl, ve) != (-1, -1):
                    raise ValueError('victim supplied for an empty slot')
            elif victim != (vl, ve) or victim in pinned:
                raise ValueError('victim/slot/pinning mismatch')
            local[rank].miss_ops.append((layer, expert, slot, vl, ve, order))
    admission = AdmissionResult(assignment, quotas, POLICIES[policy])
    if isinstance(controller.admission, FrozenRankDemandOracle):
        ctx = AdmissionContext(layer, sorted(assignment), routes.origin_ranks, effective,
                               preowned, cache, world, controller.affinity)
        if controller.admission.place(ctx) != admission:
            raise ValueError('frozen oracle decision mismatch')
    if routes.full_router_probs is not None:
        controller.history.update(layer, routes.full_router_probs)
    assert cache.next_tick() == tick
    for expert in sorted(assignment):
        rank = assignment[expert]
        slot, vl, ve, order = map(int, rows[expert, 5:])
        if vl >= 0:
            if cache.evict((vl, ve)) != (rank, slot):
                raise AssertionError('validated victim changed during plan application')
        cache.place(rank, (layer, expert), slot, tick)
    for expert in owners:
        cache.touch((layer, expert), tick)
    cache.assert_consistent()
    return LayerPlan(layer, substitution, admission, owners, local, effective, pinned)
