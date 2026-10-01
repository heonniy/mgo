#!/usr/bin/env python3
"""Audit all rank receipts and emit compact model-matrix evidence tables."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--native-reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--partial", action="store_true")
    p.add_argument("--hidden-size", type=int, default=2048)
    args = p.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    native = {row["sample_id"]: row for row in json.loads((Path(args.native_reference) / "samples.json").read_text())}
    cells, missing, audits = [], [], []
    for phase in ("ablation", "matrix"):
        manifest = json.loads((root / f"{phase}_cells.json").read_text())
        for world in (4, 8):
            folder = root / f"{phase}_r{world}"
            provenance_path = folder / "provenance-rank0.json"
            if not provenance_path.exists():
                missing.append(str(provenance_path))
                continue
            provenance = json.loads(provenance_path.read_text())
            for cell in manifest:
                repeats, first_samples = [], None
                first_metrics = None
                for repeat in range(provenance["repeats"]):
                    paths = [folder / f"{cell['name']}-rep{repeat}-rank{rank}.json" for rank in range(world)]
                    absent = [str(path) for path in paths if not path.exists()]
                    if absent:
                        missing.extend(absent)
                        continue
                    ranks = [json.loads(path.read_text()) for path in paths]
                    first = ranks[0]
                    label = f"{phase}/R{world}/{cell['name']}/rep{repeat}"
                    assert all(row["status"] == "PASS" for row in ranks), label
                    assert all(row["rank"] == rank and row["world"] == world for rank, row in enumerate(ranks)), label
                    assert all(row["metrics"] == first["metrics"] for row in ranks), label + ": replicated policy drift"
                    assert sum(row["cache_stats"][3] for row in ranks) == first["metrics"]["fetches"], label + ": fetch accounting"
                    samples = [sample for row in ranks for sample in row["quality"]["samples"]]
                    sample_signature = [(row["sample_id"], row["token_ids"]) for row in samples]
                    if first_samples is not None:
                        assert sample_signature == first_samples, label + ": repeat token mismatch"
                        assert first["metrics"] == first_metrics, label + ": repeat policy mismatch"
                    first_samples, first_metrics = sample_signature, first["metrics"]
                    assert len({row["sample_id"] for row in samples}) == world * cell["batch"], label + ": duplicate questions"
                    prefill = first["prefill"]["metrics"]
                    decode = {key: first["metrics"][key] - prefill[key]
                              for key in ("hit", "subhit", "miss", "fetches", "reloads", "remote_token_rank_pairs",
                                          "substituted_gate_mass", "total_gate_mass", "rank_token_cv_sum", "events")}
                    denom = max(1, decode["hit"] + decode["subhit"] + decode["miss"])
                    values = {"ttft_seconds": first["ttft_seconds"], "tpot_seconds": first["tpot_seconds"],
                              "generation_seconds": first["global_max_generation_seconds"],
                              "output_tokens_per_second": first["fixed_step_output_tokens_per_second"],
                              "controller_seconds_max_rank": max(row["controller_seconds"] for row in ranks),
                              "decode_controller_seconds_max_rank": max(row["controller_seconds"] - row["prefill"]["controller_seconds"] for row in ranks),
                              "host_fetch_bytes": sum(row["host_fetch_bytes"] for row in ranks),
                              "decode_fetches": decode["fetches"], "decode_reloads": decode["reloads"],
                              "decode_remote_token_rank_pairs": decode["remote_token_rank_pairs"],
                              "decode_hit_pct": 100 * decode["hit"] / denom,
                              "decode_subhit_pct": 100 * decode["subhit"] / denom,
                              "decode_miss_pct": 100 * decode["miss"] / denom,
                              "decode_substituted_gate_mass_pct": 100 * decode["substituted_gate_mass"] / max(1e-30, decode["total_gate_mass"]),
                              "decode_mean_rank_token_cv": decode["rank_token_cv_sum"] / max(1, decode["events"]),
                              "quality_correct": sum(row["correct"] for row in samples), "quality_total": len(samples),
                              "quality_unfinished": sum(not row["finished"] for row in samples),
                              "native_correct_same_questions": sum(native[row["sample_id"]]["correct"] for row in samples),
                              "native_unfinished_same_questions": sum(not native[row["sample_id"]]["finished"] for row in samples),
                              "native_token_mismatch_samples": sum(row["token_ids"] != native[row["sample_id"]]["token_ids"] for row in samples),
                              "quality_gains_vs_native": sum(row["correct"] and not native[row["sample_id"]]["correct"] for row in samples),
                              "quality_losses_vs_native": sum(not row["correct"] and native[row["sample_id"]]["correct"] for row in samples)}
                    for key in ("peer_payload_tx_bytes", "collective_cuda_ms_max_rank", "decode_peer_payload_tx_bytes"):
                        values[key] = None
                    if "collectives" in first:
                        transmitted = sum(row["collectives"]["peer_payload_tx_bytes"].get("dispatch_hidden", 0) for row in ranks)
                        expected = first["metrics"]["remote_token_rank_pairs"] * args.hidden_size * 2
                        assert transmitted == expected, label + ": deduplicated dispatch byte accounting"
                        values["peer_payload_tx_bytes"] = sum(sum(row["collectives"]["peer_payload_tx_bytes"].values()) for row in ranks)
                        values["decode_peer_payload_tx_bytes"] = values["peer_payload_tx_bytes"] - sum(sum(row["prefill"]["peer_payload_tx_bytes"].values()) for row in ranks)
                        values["collective_cuda_ms_max_rank"] = max(sum(row["collectives"]["cuda_interval_ms"].values()) for row in ranks)
                    repeats.append(values)
                    audits.append({"run": label, "status": "PASS", "rank_receipts": [str(path) for path in paths]})
                if len(repeats) != provenance["repeats"]:
                    continue
                result = {"phase": phase, "world": world, "cell": cell["name"], "local_batch": cell["batch"],
                          "cache_ratio": cell["config"]["global_cache_ratio"], "eviction": cell["config"]["eviction"],
                          "admission": cell["config"]["admission"], "instrumented": provenance["instrumented"],
                          "repeats": len(repeats), "steps": provenance["steps"]}
                result.update({key: median(repeats, key) if repeats[0][key] is not None else None for key in repeats[0]})
                result["repeat_measurements"] = repeats
                cells.append(result)
    result = {"status": "PARTIAL" if missing else "PASS", "completed_cells": len(cells), "missing_receipts": missing,
              "audits": audits, "cells": cells,
              "scope": "Medians of recorded repeats; cold-cache fixed-step generation and short numeric quality screen. No confidence intervals or quality-safety claim."}
    (output / "matrix_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for phase in ("ablation", "matrix"):
        selected = [{key:value for key,value in row.items() if key != "repeat_measurements"} for row in cells if row["phase"] == phase]
        if selected:
            with (output / f"{phase}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(selected[0]))
                writer.writeheader()
                writer.writerows(selected)
    print(json.dumps({"status": result["status"], "completed_cells": len(cells), "audited_repeats": len(audits), "missing": len(missing)}))
    if missing and not args.partial:
        raise AssertionError("matrix is incomplete; use --partial only for progress inspection")


if __name__ == "__main__":
    main()
