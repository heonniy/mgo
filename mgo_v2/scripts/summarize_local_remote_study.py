#!/usr/bin/env python3
"""Audit every rank/repeat before publishing the frozen locality study."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


EXPERT_BYTES = 9437184


def write_table(root, name, rows):
    if not rows:
        return
    with (root / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def payload(metrics):
    return (metrics["remote_token_rank_pairs"] * (2048 * 2 + 8 + 8 * 10)
            + metrics["remote_expert_routes"] * (2048 * 2 + 16))


def control_payload(world, batch, prefill_max_rows, decode_steps=64):
    # Per event: router IDs(int64), weights(BF16), probabilities(fp32),
    # one scalar-count all-gather and two world-length count all-gathers.
    peer_copies = world * (world - 1)
    per_event_counts = 8 + 16 * world
    decode = 48 * peer_copies * decode_steps * (batch * 592 + per_event_counts)
    prefill = 48 * peer_copies * (prefill_max_rows * 592 + per_event_counts)
    return prefill + decode, decode


def summarize(root, output, partial=False):
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((root / "status.json").read_text())
    package = Path(__file__).resolve().parents[1]
    binding = json.loads((package / "experiments/local_remote_e2e_impact_20261001/measurement_manifest.json").read_text())
    extended = bool(binding.get("user_requested_extension"))
    gpu_maps = binding.get("physical_gpus_by_world", {"8": list(range(8)), "4": list(range(4))})
    def check_boot(boot, world, rank):
        assert boot["local_rank"] == rank
        assert boot["visible_gpu"] == str(gpu_maps[str(world)][rank]), "physical GPU selection mismatch"
        assert boot["strict_numa"] and boot["numa_policy"] == "membind-strict"
    for name, digest in state["source_hashes"].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, "source changed during study: " + name
    extension, = (package.parent / "MoE-Infinity-EP-archer-coslot/moe_infinity").glob("_store*.so")
    assert hashlib.sha256(extension.read_bytes()).hexdigest() == binding["extension_sha256"]
    fingerprint = hashlib.sha256()
    for source in sorted((package / "mgo_v2").glob("*.py")) + [package / "examples/benchmark_model.py"]:
        fingerprint.update(source.name.encode())
        fingerprint.update(source.read_bytes())
    fingerprint.update(extension.read_bytes())
    expected_fingerprint = fingerprint.hexdigest()
    evidence = {}
    def read(path):
        evidence[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    a_rows, a_iterations, b_rows, rank_rows, completeness = [], [], [], [], []
    for world in (8, 4):
        directory = root / f"stage_a_r{world}"
        paths = [directory / f"rank{rank}.json" for rank in range(world)]
        if not all(p.exists() for p in paths):
            if not partial:
                raise AssertionError(f"incomplete Stage A/R{world}")
        else:
            ranks = [read(p) for p in paths]
            selected_events = read(directory / "selection.json")
            assert selected_events["pool"] == 384
            for rank, receipt in enumerate(ranks):
                assert receipt["status"] == "PASS" and receipt["rank"] == rank and receipt["world"] == world
                check_boot(receipt["boot"], world, rank)
                assert receipt["counter_oracle_events"] == 432
                assert receipt["checkpoint"] == binding["validated_checkpoint_identity"]
                assert len(receipt["results"]) == 15
                assert receipt["selections"] == ranks[0]["selections"]
                assert receipt["selections"] == selected_events["events"]
                event_path = directory / f"events-rank{rank}.pt"
                evidence[str(event_path.relative_to(root))] = hashlib.sha256(event_path.read_bytes()).hexdigest()
                for name, digest in receipt["source_hashes"].items():
                    assert digest == state["source_hashes"][name], name
            for index, cell in enumerate(ranks[0]["results"]):
                cells = [rank["results"][index] for rank in ranks]
                for entry in cells:
                    assert entry["mode"] == cell["mode"] and entry["event"] == cell["event"]
                    assert entry["owners"] == cell["owners"] and entry["quotas"] == cell["quotas"]
                    assert entry["accounting"] == cell["accounting"]
                    assert entry["bitwise_output_equal"] and entry["timed_expert_h2d_bytes"] == 0
                    assert len(entry["records"]) == 40
                    assert all(record["expert_fetches"] == 0 for record in entry["records"])
                timings, diagnostic_intervals = [], []
                for iteration in range(20):
                    unprofiled = [entry["records"][iteration] for entry in cells]
                    profiled = [entry["records"][20 + iteration] for entry in cells]
                    assert all(not row["instrumented"] and row["iteration"] == iteration for row in unprofiled)
                    assert all(row["instrumented"] and row["iteration"] == iteration for row in profiled)
                    maximum = max(row["layer_seconds"] for row in unprofiled)
                    kinds = ("dispatch_hidden", "dispatch_metadata", "return_outputs", "return_metadata")
                    actual_payload = sum(sum(row["collectives"]["peer_payload_tx_bytes"].get(k, 0) for k in kinds)
                                         for row in profiled)
                    assert actual_payload == payload(cell["accounting"]), (world, cell["mode"], actual_payload)
                    count_bytes = sum(row["collectives"]["peer_payload_tx_bytes"].get("counts", 0) for row in profiled)
                    assert count_bytes == 16 * world * world * (world - 1)
                    interval = max(sum(row["collectives"]["cuda_interval_ms"].values())
                                   for row in profiled)
                    timings.append(maximum)
                    diagnostic_intervals.append(interval)
                    a_iterations.append(dict(world=world, event=cell["event"], mode=cell["mode"], iteration=iteration,
                        remote_pair_fraction=cell["accounting"]["remote_pair_fraction"],
                        moe_layer_seconds=maximum, diagnostic_nccl_interval_ms=interval,
                        peer_payload_tx_bytes=actual_payload, all_submitted_peer_tx_bytes=actual_payload + count_bytes,
                        expert_h2d_bytes=0))
                counts = cell["accounting"]
                a_rows.append(dict(world=world, event=cell["event"], mode=cell["mode"],
                    layer=cell["selection"]["layer"], active_experts=cell["selection"]["active"],
                    local_pair_fraction=counts["local_pair_fraction"], remote_pair_fraction=counts["remote_pair_fraction"],
                    route_local_fraction=counts["route_local_fraction"], local_pairs=counts["local_token_rank_pairs"],
                    remote_pairs=counts["remote_token_rank_pairs"], rank_token_cv=counts["rank_token_cv"],
                    rank_token_max_mean=counts["rank_token_max_mean"],
                    peer_payload_tx_bytes=payload(counts),
                    all_submitted_peer_tx_bytes=payload(counts) + 16 * world * world * (world - 1), expert_h2d_bytes=0,
                    moe_seconds_median=statistics.median(timings), moe_seconds_min=min(timings), moe_seconds_max=max(timings),
                    diagnostic_nccl_ms_median=statistics.median(diagnostic_intervals),
                    diagnostic_nccl_ms_min=min(diagnostic_intervals), diagnostic_nccl_ms_max=max(diagnostic_intervals)))
            completeness.append(dict(stage="A", world=world, cells=15, iterations=300, diagnostic_iterations=300))

        directory = root / f"stage_b_r{world}"
        manifest = read(root / f"cells_r{world}.json")
        jobs = [(directory, root / f"cells_r{world}.json")]
        directories = {cell["name"]: directory for cell in manifest}
        if extended and world == 8:
            extra = read(root / "cells_r8_extended.json")
            assert len(extra) == 6 and {cell["batch"] for cell in extra} == {16, 32}
            extra_directory = root / "stage_b_r8_extended"
            directories.update({cell["name"]: extra_directory for cell in extra})
            manifest += extra
            jobs.append((extra_directory, root / "cells_r8_extended.json"))
        for cell in manifest:
            directory = directories[cell["name"]]
            reference_by_rank = {}
            for repeat in range(5):
                paths = [directory / f"{cell['name']}-rep{repeat}-rank{rank}.json" for rank in range(world)]
                if not all(path.exists() for path in paths):
                    if partial:
                        continue
                    raise AssertionError(f"incomplete {cell['name']}/R{world}/repeat{repeat}")
                records = [read(path) for path in paths]
                for rank, row in enumerate(records):
                    assert row["status"] == "PASS" and row["world"] == world and row["rank"] == rank
                    assert row["name"] == cell["name"]
                    assert row["repeat"] == repeat and row["local_batch"] == cell["batch"]
                    assert "collectives" not in row, "instrumented timing cannot enter primary results"
                    for key, value in cell["config"].items():
                        assert row["config"][key] == value, (cell["name"], key)
                    assert row["config"]["same_layer_alpha"] == .25 and row["config"]["path_eta"] == .5
                    assert len(row["rank_step_seconds"]) == 65
                    assert len(row["generated_token_ids"]) == len(row["prompt_tokens"]) == cell["batch"]
                    assert all(len(ids) == 65 for ids in row["generated_token_ids"])
                    assert row["metrics"] == records[0]["metrics"]
                    assert row["prefill"]["metrics"] == records[0]["prefill"]["metrics"]
                    assert row["host_fetch_bytes"] == row["cache_stats"][3] * EXPERT_BYTES
                    assert row["metrics"]["events"] == 65 * 48
                    semantic = {key: row[key] for key in ("metrics", "cache_stats", "fetch_modes", "generated_token_ids", "quality")}
                    if rank in reference_by_rank:
                        assert semantic == reference_by_rank[rank], (world, cell["name"], repeat, rank, "nondeterministic")
                    else:
                        reference_by_rank[rank] = semantic
                    rank_rows.append(dict(world=world, cell=cell["name"], repeat=repeat, rank=rank,
                        controller_seconds=row["controller_seconds"], generation_seconds=row["rank_generation_seconds"],
                        physical_fetches=row["cache_stats"][3], host_fetch_bytes=row["host_fetch_bytes"],
                        source=str(paths[rank].relative_to(root))))
                first = records[0]
                metrics = first["metrics"]
                prefill = first["prefill"]["metrics"]
                count_keys = ("local_token_rank_pairs", "remote_token_rank_pairs", "local_expert_routes", "remote_expert_routes",
                              "fetches", "reloads", "rank_token_cv_sum", "rank_token_max_mean_sum", "events")
                decode = {key: metrics[key] - prefill[key] for key in count_keys}
                assert sum(r["cache_stats"][3] for r in records) == metrics["fetches"]
                assert sum(r["prefill"]["cache_stats"][3] for r in records) == prefill["fetches"]
                max_steps = [max(row["rank_step_seconds"][i] for row in records) for i in range(65)]
                assert first["global_max_step_seconds"] == max_steps
                assert first["ttft_seconds"] == max_steps[0]
                assert first["tpot_seconds"] == sum(max_steps[1:]) / 64
                assert first["global_max_generation_seconds"] == max(row["rank_generation_seconds"] for row in records)
                assert first["fixed_step_output_tokens_per_second"] == world * cell["batch"] * 65 / first["global_max_generation_seconds"]
                for row in records:
                    for key in ("global_max_step_seconds", "ttft_seconds", "tpot_seconds",
                                "global_max_generation_seconds", "fixed_step_output_tokens_per_second"):
                        assert row[key] == first[key], (world, cell["name"], repeat, key)
                control, decode_control = control_payload(world, cell["batch"], max(sum(row["prompt_tokens"]) for row in records))
                b_rows.append(dict(world=world, local_batch=cell["batch"], global_batch=world * cell["batch"],
                    policy=cell["config"]["admission"], cell=cell["name"], repeat=repeat,
                    ttft_seconds=first["ttft_seconds"], tpot_seconds=first["tpot_seconds"],
                    generation_seconds=first["global_max_generation_seconds"],
                    output_tokens_per_second=first["fixed_step_output_tokens_per_second"],
                    controller_seconds=max(row["controller_seconds"] for row in records),
                    decode_controller_seconds=max(row["controller_seconds"] - row["prefill"]["controller_seconds"] for row in records),
                    physical_expert_h2d_bytes=metrics["fetches"] * EXPERT_BYTES,
                    decode_expert_h2d_bytes=decode["fetches"] * EXPERT_BYTES,
                    fetches=metrics["fetches"], decode_fetches=decode["fetches"], reloads=metrics["reloads"],
                    decode_reloads=decode["reloads"], local_pair_fraction=metrics["local_pair_fraction"],
                    remote_pair_fraction=metrics["remote_pair_fraction"], route_local_fraction=metrics["route_local_fraction"],
                    local_pairs=metrics["local_token_rank_pairs"], remote_pairs=metrics["remote_token_rank_pairs"],
                    decode_remote_pair_fraction=decode["remote_token_rank_pairs"] / (decode["remote_token_rank_pairs"] + decode["local_token_rank_pairs"]),
                    decode_remote_pairs=decode["remote_token_rank_pairs"], peer_payload_tx_bytes=payload(metrics),
                    routing_and_count_tx_bytes=control, all_submitted_peer_tx_bytes=payload(metrics) + control,
                    decode_all_submitted_peer_tx_bytes=payload(decode) + decode_control,
                    decode_peer_payload_tx_bytes=payload(decode), rank_token_cv=metrics["mean_event_rank_token_cv"],
                    decode_rank_token_cv=decode["rank_token_cv_sum"] / decode["events"],
                    rank_token_max_mean=metrics["mean_event_rank_token_max_mean"]))
            completeness.append(dict(stage="B", world=world, cell=cell["name"], repeats=len(reference_by_rank) and
                                     sum(row["world"] == world and row["cell"] == cell["name"] for row in b_rows)))
        for directory, manifest_path in jobs:
            provenance_paths = [directory / f"provenance-rank{rank}.json" for rank in range(world)]
            if not partial:
                assert all(path.exists() for path in provenance_paths), f"missing provenance: {directory}"
            for rank, path in enumerate(provenance_paths):
                if path.exists():
                    provenance = read(path)
                    assert provenance["world"] == world and provenance["rank"] == rank
                    check_boot(provenance["boot"], world, rank)
                    assert not provenance["instrumented"] and provenance["steps"] == 65 and provenance["repeats"] == 5
                    assert provenance["code_sha256"] == expected_fingerprint
                    assert provenance["checkpoint"] == binding["validated_checkpoint_identity"]
                    assert provenance["versions"] == binding["expected_versions"]
                    for input_path, digest in binding["input_hashes"].items():
                        key = {"screen_workload.json": "workload", "similarity.npy": "similarity", "affinity.npz": "affinity"}[Path(input_path).name]
                        assert provenance["input_sha256"][key] == digest
                    assert provenance["input_sha256"]["cells"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not partial:
        assert len(b_rows) == (75 if extended else 45) and len(a_rows) == 30
        assert len(rank_rows) == (540 if extended else 300)
        assert state["status"] == "TIMING_COMPLETE"
        names = [run["name"] for run in state["runs"]]
        original_order = ["stage_a_r8", "stage_b_r8", "stage_a_r4", "stage_b_r4"]
        integrated_order = ["stage_a_r8", "stage_b_r8", "stage_b_r8_extended", "stage_a_r4", "stage_b_r4"]
        assert names == original_order or (extended and names == integrated_order)
        assert all(run["exit_code"] == 0 for run in state["runs"])
        if binding.get("user_requested_r4_devices"):
            for run in state["runs"]:
                if run["name"].endswith("r4"):
                    assert run["physical_gpus"] == gpu_maps["4"]
            change = read(root / "r4_device_change.json")
            assert change["physical_gpus"] == gpu_maps["4"]
        if extended and names == original_order:
            extension = read(root / "extension_status.json")
            pause = read(root / "scheduling_pause.json")
            assert extension["status"] == "PASS" and extension["exit_code"] == 0
            assert pause["status"] == "RESUMED_AFTER_R8_EXTENSION"
            assert extension["finished_unix"] <= state["runs"][2]["started_unix"]
            assert extension["original_r8_process_finished_unix"] <= extension["started_unix"]
    write_table(output, "local_remote_sensitivity", a_rows)
    write_table(output, "local_remote_iterations", a_iterations)
    write_table(output, "e2e_repeats", b_rows)
    write_table(output, "e2e_rank_receipts", rank_rows)
    (output / "local_remote_sensitivity.json").write_text(json.dumps(dict(rows=a_rows, iterations=a_iterations), indent=2) + "\n")
    (output / "e2e_repeats.json").write_text(json.dumps(dict(rows=b_rows, ranks=rank_rows), indent=2) + "\n")
    audit = dict(status="PARTIAL" if partial else "PASS", stage_a_cells=len(a_rows), stage_b_repeats=len(b_rows),
                 user_requested_extended_batches=extended,
                 runtime_fingerprint=expected_fingerprint, extension_sha256=binding["extension_sha256"],
                 physical_gpus_by_world=gpu_maps,
                 rank_receipts=len(rank_rows), completeness=completeness, source_sha256=evidence,
                 scopes={"peer_payload": "Submitted dispatch+native-order-return tensors, excludes routing/count collectives and wire overhead",
                         "all_submitted_peer": "Dispatch+return plus router/count all-gather tensor copies; excludes wire overhead",
                         "h2d": "Physical dispatcher fetch accounting; actual trace observation is in profile_summary",
                         "stage_a": "Uninstrumented resident layer latency; separate CUDA-event diagnostic intervals include waits"})
    (output / "validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("status", "stage_a_cells", "stage_b_repeats", "rank_receipts")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    summarize(Path(args.root), Path(args.output), args.partial)
