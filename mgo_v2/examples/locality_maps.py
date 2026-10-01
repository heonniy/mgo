"""Deterministic quota-preserving maps, selected without timing feedback."""
import numpy as np


MODES = ("LOCAL_MAX", "LOCAL_HIGH", "LOCAL_MID", "LOCAL_LOW", "REMOTE_MAX")


def accounting(origins, routes, owners, world):
    """Independent Boolean incidence oracle (deduplicates token destinations)."""
    incidence = np.zeros((len(routes), world), dtype=bool)
    local_routes = remote_routes = 0
    for t, experts in enumerate(routes):
        for expert in experts:
            destination = owners[int(expert)]
            incidence[t, destination] = True
            if destination == int(origins[t]):
                local_routes += 1
            else:
                remote_routes += 1
    local = int(incidence[np.arange(len(routes)), origins].sum())
    remote = int(incidence.sum()) - local
    loads = incidence.sum(0)
    return dict(local_token_rank_pairs=local, remote_token_rank_pairs=remote,
                local_expert_routes=local_routes, remote_expert_routes=remote_routes,
                remote_pair_fraction=remote / max(local + remote, 1),
                local_pair_fraction=local / max(local + remote, 1),
                route_local_fraction=local_routes / max(local_routes + remote_routes, 1),
                rank_token_loads=loads.tolist(),
                rank_token_cv=float(loads.std() / max(loads.mean(), 1)),
                rank_token_max_mean=float(loads.max() / max(loads.mean(), 1)))


def make_maps(origins, selected, world, trials=20000, seed=20261001):
    experts = sorted(set(int(e) for e in selected.reshape(-1)))
    quotas = [len(experts) // world + (r < len(experts) % world) for r in range(world)]
    initial = np.repeat(np.arange(world), quotas)
    np.random.default_rng(seed).shuffle(initial)
    membership = np.stack([(selected == e).any(1) for e in experts], axis=1).astype(np.int16)
    proposals = np.random.default_rng(seed + 1).integers(len(experts), size=(trials, 2))
    remote_mask = np.arange(world)[None, :] != origins[:, None]

    def solve(target):
        owners = initial.copy()
        counts = np.stack([membership[:, owners == r].sum(1) for r in range(world)], axis=1)
        current = int(((counts > 0) & remote_mask).sum())
        accepted = 0
        for a, b in proposals:
            ra, rb = owners[a], owners[b]
            if ra == rb:
                continue
            delta = membership[:, b] - membership[:, a]
            ca, cb = counts[:, ra] + delta, counts[:, rb] - delta
            next_value = current + int((((ca > 0).astype(int) - (counts[:, ra] > 0)) * remote_mask[:, ra]).sum())
            next_value += int((((cb > 0).astype(int) - (counts[:, rb] > 0)) * remote_mask[:, rb]).sum())
            if abs(next_value - target) < abs(current - target):
                owners[a], owners[b] = rb, ra
                counts[:, ra], counts[:, rb] = ca, cb
                current = next_value
                accepted += 1
        mapping = dict(zip(experts, map(int, owners)))
        assert accounting(origins, selected, mapping, world)["remote_token_rank_pairs"] == current
        assert np.bincount(owners, minlength=world).tolist() == quotas
        return mapping, current, accepted

    low = solve(0)
    high = solve(len(selected) * min(world - 1, selected.shape[1]))
    targets = np.linspace(low[1], high[1], 5)
    solutions = [low] + [solve(float(t)) for t in targets[1:4]] + [high]
    return [{"mode": mode, "owners": mapping, "quotas": quotas, "remote_target": float(target),
             "accepted_swaps": accepted, "swap_trials": trials,
             "accounting": accounting(origins, selected, mapping, world)}
            for mode, target, (mapping, _, accepted) in zip(MODES, targets, solutions)]
