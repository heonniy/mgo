"""Summarize a bounded native frozen-route policy rank diagnosis."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    status = read(root / "status.json")
    assert status["status"] == "PASS"
    result = read(root / "result.json")
    assert result["policy_compare"] and result["prefetch"] == "off"
    pair = tuple(result.get("policy_pair") or ("BR", "LA_CA_NEAR"))
    assert len(pair) == 2 and len(set(pair)) == 2
    primary = defaultdict(list)
    for run in range(4):
        row = read(root / f"run{run}.json")
        assert row["status"] == "PASS"
        primary[row["backend"]].append(row)
    assert set(primary) == set(pair)
    assert all(len(rows) == 2 for rows in primary.values())
    policies = {}
    for policy in pair:
        ranks = []
        for rank in range(4):
            d = read(root / f"diagnostic_{policy}_rank{rank}.json")
            assert d["status"] == "PASS" and d["token_prefix_parity"]
            assert d["policy"] == policy and d["decode_steps"] == 16
            phases = defaultdict(float)
            cpu = defaultdict(float)
            for segment in d["segments"]:
                if segment["event_index"] >= 48:
                    phases[segment["phase"]] += segment["stream_seconds"]
                    cpu[segment["phase"]] += segment["cpu_ns"] / 1e9
            events = d["decode_cache_events"]
            assert len(events) == 16 * 48
            counts = {key: sum(e[key] for e in events) for key in (
                "expert_uses", "token_expert_uses", "main_hits", "prefetch_hits",
                "demand_misses", "main_hit_tokens", "miss_tokens")}
            assert counts["prefetch_hits"] == 0
            per_step = {str(step): {
                "expert_groups": sum(e["expert_uses"] for e in events if e["step"] == step),
                "expert_token_rows": sum(e["token_expert_uses"] for e in events if e["step"] == step),
                "mandatory_copies": sum(e["demand_misses"] for e in events if e["step"] == step),
            } for step in range(1, 17)}
            primary_rank = read(root / f"run{pair.index(policy)}_rank{rank}.json")
            ranks.append({
                "rank": rank, "gpu": (0, 1, 4, 5)[rank],
                "diagnostic_wall_s": d["wall_seconds"],
                "decode_phase_stream_s_per_token": {key: value / 16 for key, value in phases.items()},
                "decode_phase_cpu_s_per_token": {key: value / 16 for key, value in cpu.items()},
                "prefix_counts": counts, "per_step": per_step,
                "primary_decode_peer_bytes_63": sum(primary_rank["decode_bytes"][k] for k in ("forward", "returned")),
                "primary_decode_h2d_bytes_63": primary_rank["decode_bytes"]["h2d"],
            })
        policies[policy] = {
            "primary": primary[policy],
            "primary_mean_tpot_s": sum(row["TPOT"] for row in primary[policy]) / 2,
            "primary_mean_ttft_s": sum(row["TTFT"] for row in primary[policy]) / 2,
            "primary_mean_e2e_s": sum(row["E2E"] for row in primary[policy]) / 2,
            "diagnostic_ranks": ranks,
            "prefix_total_expert_rows": sum(r["prefix_counts"]["token_expert_uses"] for r in ranks),
            "prefix_total_expert_groups": sum(r["prefix_counts"]["expert_uses"] for r in ranks),
            "prefix_total_mandatory_copies": sum(r["prefix_counts"]["demand_misses"] for r in ranks),
            "primary_decode_peer_bytes_63": sum(r["primary_decode_peer_bytes_63"] for r in ranks),
            "primary_decode_h2d_bytes_63": sum(r["primary_decode_h2d_bytes_63"] for r in ranks),
        }
    assert policies[pair[0]]["prefix_total_expert_rows"] == policies[pair[1]]["prefix_total_expert_rows"]
    raw_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(root.glob("diagnostic_*_rank*.json"))}
    return {
        "status": "PASS", "source": str(root), "cell": result["cell"],
        "source_commit": status["source_commit"], "raw_diagnostic_sha256": raw_hashes,
        "primary_decode_steps": result["decode_steps"], "diagnostic_decode_prefix": 16,
        "policy_pair": list(pair),
        "route_frozen": result["route_frozen"], "prefetch": result["prefetch"],
        "scope": "Uninstrumented two-repeat primary uses 63 decode forwards. One separately instrumented 16-forward frozen prefix per policy; stream spans include host gaps and peer waits and must not replace TPOT.",
        "policies": policies,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(summarize(a.root), indent=2) + "\n")
