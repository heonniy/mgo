"""Reconcile per-layer CA/Near expert and return spans from existing diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EVENTS = 16 * 48
POLICIES = ("LA_CA_NEAR", "CA")


def load_policy(root, policy):
    names = ("moe.expert_compute", "return_token_a2a", "placement_controller_cpu")
    spans = {name: np.zeros((EVENTS, 4), np.float64) for name in names}
    calls = {name: np.zeros((EVENTS, 4), np.int32) for name in names}
    return_cpu = np.zeros((EVENTS, 4), np.float64)
    rows = np.zeros((EVENTS, 4), np.int32)
    groups = np.zeros((EVENTS, 4), np.int32)
    hashes = {}
    for rank in range(4):
        path = root / f"diagnostic_{policy}_rank{rank}.json"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        assert data["status"] == "PASS" and data["token_prefix_parity"]
        for segment in data["segments"]:
            event = segment["event_index"] - 48
            if not 0 <= event < EVENTS:
                continue
            name = segment["phase"]
            if name in spans:
                spans[name][event, rank] += segment["stream_seconds"] * 1000
                calls[name][event, rank] += 1
                if name == "return_token_a2a":
                    return_cpu[event, rank] += segment["cpu_ns"] / 1e6
        for event in data["decode_cache_events"]:
            index = event["event_index"] - 48
            rows[index, rank] = event["token_expert_uses"]
            groups[index, rank] = event["expert_uses"]
    assert all(np.all(count == 1) for count in calls.values())
    assert np.all(rows.sum(axis=1) == 512)
    expert = spans["moe.expert_compute"]
    returned = spans["return_token_a2a"]
    correlations = []
    group_correlations = []
    row_correlations = []
    for event in range(EVENTS):
        if expert[event].std() > 0 and returned[event].std() > 0:
            correlations.append(float(np.corrcoef(expert[event], returned[event])[0, 1]))
        if expert[event].std() > 0 and groups[event].std() > 0:
            group_correlations.append(float(np.corrcoef(groups[event], expert[event])[0, 1]))
        if expert[event].std() > 0 and rows[event].std() > 0:
            row_correlations.append(float(np.corrcoef(rows[event], expert[event])[0, 1]))
    return dict(
        raw_sha256=hashes,
        expert_max_ms_per_layer=float(expert.max(axis=1).mean()),
        expert_min_ms_per_layer=float(expert.min(axis=1).mean()),
        expert_rank_spread_ms_per_layer=float(np.ptp(expert, axis=1).mean()),
        expert_rank_total_max_ms_per_token=float(expert.sum(axis=0).max() / 16),
        return_rank_mean_ms_per_layer=float(returned.mean()),
        return_rank_min_ms_per_layer=float(returned.min(axis=1).mean()),
        return_rank_max_ms_per_layer=float(returned.max(axis=1).mean()),
        return_rank_spread_ms_per_layer=float(np.ptp(returned, axis=1).mean()),
        return_cpu_mean_ms_per_layer=float(return_cpu.mean()),
        controller_rank_mean_ms_per_layer=float(spans["placement_controller_cpu"].mean()),
        expert_rows_max_per_layer=float(rows.max(axis=1).mean()),
        expert_rows_spread_per_layer=float(np.ptp(rows, axis=1).mean()),
        expert_groups_max_per_layer=float(groups.max(axis=1).mean()),
        expert_groups_spread_per_layer=float(np.ptp(groups, axis=1).mean()),
        within_layer_group_expert_correlation_median=float(np.median(group_correlations)),
        within_layer_rows_expert_correlation_median=float(np.median(row_correlations)),
        within_layer_expert_return_correlation_median=float(np.median(correlations)),
        within_layer_expert_return_negative_fraction=float(np.mean(np.array(correlations) < 0)),
        slowest_expert_fastest_return_fraction=float(np.mean(expert.argmax(axis=1) == returned.argmin(axis=1))),
        fastest_expert_slowest_return_fraction=float(np.mean(expert.argmin(axis=1) == returned.argmax(axis=1))),
        critical_expert_rank_switch_fraction=float(np.mean(expert.argmax(axis=1)[1:] != expert.argmax(axis=1)[:-1])),
        _expert_max=expert.max(axis=1),
        _return_mean=returned.mean(axis=1),
        _groups_max=groups.max(axis=1),
    )


def main(root):
    near, ca = (load_policy(root, policy) for policy in POLICIES)
    cross = dict(
        matched_layer_delta_expert_max_to_delta_return_mean_pearson=float(
            np.corrcoef(ca["_expert_max"] - near["_expert_max"],
                        ca["_return_mean"] - near["_return_mean"])[0, 1]),
        matched_layer_delta_group_max_to_delta_expert_max_pearson=float(
            np.corrcoef(ca["_groups_max"] - near["_groups_max"],
                        ca["_expert_max"] - near["_expert_max"])[0, 1]),
    )
    for policy in (near, ca):
        for key in list(policy):
            if key.startswith("_"):
                del policy[key]
    return dict(
        status="PASS", source=str(root), policy_pair=list(POLICIES),
        scope="Same frozen route, 16 decode steps x 48 layers. One separately instrumented pass per policy; CUDA current-stream spans include host submission and peer waits. Correlations are descriptive over matched layer events, not independent samples or causal coefficients.",
        per_policy={"LA_CA_NEAR": near, "CA": ca}, comparison=cross,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(main(args.root), indent=2) + "\n")
