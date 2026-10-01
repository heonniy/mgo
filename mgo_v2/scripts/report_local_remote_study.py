#!/usr/bin/env python3
"""Render the final report only after primary and posthoc audits pass."""
import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    return list(csv.DictReader(path.open()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    audit = json.loads((root / "validation.json").read_text())
    profiles = json.loads((root / "profile_summary.json").read_text())
    assert audit["status"] == profiles["status"] == "PASS"
    rows = read_csv(root / "e2e_summary.csv")
    comparisons = read_csv(root / "comparisons.csv")
    stage_a = json.loads((root / "local_remote_sensitivity.json").read_text())["rows"]
    anchors = [r for r in comparisons if r["policy"] == "hungarian_same_path" and int(r["local_batch"]) == 8]
    assert len(anchors) == 2
    repeat_count, rank_count = audit["stage_b_repeats"], audit["rank_receipts"]
    passed = all(float(r["tpot_speedup"]) > 1 and float(r["remote_fraction_change"]) < 0 for r in anchors)
    outcome = ("The prespecified directional TPOT/locality gate passes in both primary anchors. "
               "This is a bounded five-repeat observation, not a significance or production-serving claim." if passed else
               "The prespecified C-policy E2E acceptance gate does not pass in both primary anchors. "
               "Do not claim a general communication-aware E2E speedup from this study.")
    text = ["# Physical local/remote TPOT and E2E results", "", outcome, "",
        "Plan commit: `1a98d10ac17557e7a9112e12ad36d07fe5d5be27`. "
        "Measurement implementation: `e8458d4`. Raw evidence: "
        "`/home/hwlee/mgo-results/local_remote_e2e_impact_20261001`.", "",
        "Execution followed the owner's R8 → R4 order. Stage A has 30 event/map cells, "
        "600 uninstrumented resident iterations and 600 separate CUDA-event diagnostic iterations. "
        f"Stage B has {len(rows)} conditions × five repeats ({repeat_count} generations, {rank_count} rank receipts). "
        "The two preselected R8/B8 posthoc profiles ran only after all primary timing completed.", "",
        "## Fixed protocol", "",
        "Qwen3-30B-A3B-Instruct-2507 BF16 on H100 NVSwitch; cache30; Coverage W128/k1/lambda2; "
        "expert-level gate .20/similarity .65 substitution; support64/alpha=.25, "
        "path support64/eta=.5; seed42; hard quotas, one residency controller, no migration/replication. "
        "Every repeat resets logical/physical expert cache and policy history. "
        "Dense weights and CUDA allocator/kernel caches stay loaded. All study GPU jobs ran sequentially.", "",
        "64 decode steps are **one prefill plus 64 decode forwards, producing 65 fixed-work tokens**. "
        "EOS ends answer scoring, while timed computation continues. TPOT is the average of "
        "max-rank decode step wall times; generation time is the max-rank continuous wall interval. "
        "This is fixed-work generation, not production serving throughput or long-horizon task quality. "
        "The R8/B4 control uses the same first 32 questions as R4/B8; R8/B8 uses 64 questions. "
        "The owner-requested R8/B16 and B32 expansion uses 128 and 256 questions, respectively, "
        "and ran before the R4 jobs with identical policy coefficients and decode length. "
        "The original B8 acceptance anchors and preselected two profiles were retained.", "",
        "[Execution binding](EXECUTION_BINDING.md) specifies selection, solver, timer and byte definitions. "
        "[Manifest](measurement_manifest.json) binds the inputs and implementation.", "",
        "## Stage B: all five repeats", "",
        "Values are median [minimum, maximum], in seconds. Every individual repeat is available in "
        "[e2e_repeats.csv](e2e_repeats.csv), with per-rank times/fetches in "
        "[e2e_rank_receipts.csv](e2e_rank_receipts.csv). Conditions were run in fixed sequential "
        "policy blocks, so temporal effects are not removed by randomized interleaving.", "",
        "| Ranks / local batch | Policy | TTFT | TPOT | Generation |",
        "|---|---|---:|---:|---:|"]
    labels = dict(random="Random", hungarian_current="Hungarian current", hungarian_same_path="Hungarian same+path")
    for row in rows:
        def span(key):
            return f"{float(row[key + '_median']):.3f} [{float(row[key + '_min']):.3f}, {float(row[key + '_max']):.3f}]"
        text.append(f"| R{row['world']} / B{row['local_batch']} | {labels[row['policy']]} | {span('ttft_seconds')} | {span('tpot_seconds')} | {span('generation_seconds')} |")
    text += ["", "![All five primary repeats](e2e_timing.png)", "",
        "## C2 versus Random", "",
        "Positive E2E/remote reduction means less time/traffic; positive fetch change means more H2D. "
        "Ratios below use condition medians. Whole-generation locality/payload includes prefill; "
        "decode-only fields are also retained in the repeat table.", "",
        "| Ranks / local batch | TPOT speedup | E2E reduction | Remote-pair reduction | Remote-fraction change | Fetch change |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        if row["policy"] != "hungarian_same_path":
            continue
        text.append(f"| R{row['world']} / B{row['local_batch']} | {float(row['tpot_speedup']):.3f}× | "
                    f"{100 * float(row['e2e_reduction']):+.2f}% | {100 * float(row['remote_reduction']):+.2f}% | "
                    f"{100 * float(row['remote_fraction_change']):+.2f} pp | {100 * float(row['fetch_change']):+.2f}% |")
    text += ["", "Do not attribute the full timing difference to NVLink: admission also changes "
        "resident experts, future substitutions, expert fetches, controller CPU work and rank load. "
        "Physical expert bytes below are dispatcher fetch accounting; the posthoc profiles independently "
        "verify actual copies for the two selected R8/B8 conditions.", "",
        "![System metric comparison](system_metrics.png)", "",
        "## Stage A: resident communication sensitivity", "",
        "The same real token states, selected expert identities, BF16 arithmetic and hard quotas "
        "are reused across all five owner maps for each event. Every output matches its captured "
        "exact-routing reference bitwise. All timed iterations have zero physical expert fetches. "
        "The unchanged direct-slot binary's earlier CUDA trace audits establish no hidden "
        "per-hit expert upload or full-expert D2D. This stage checks the actual fetch counter "
        "and residency on every iteration; it does not collect a new CUPTI transfer trace.", "",
        "Endpoints minimize/maximize exact deduplicated remote-pair **count** using fixed-budget "
        "quota-preserving swaps, not a globally optimal solver. Thus map labels do not promise "
        "monotonic fractions: the total number of local+remote pairs can change too. "
        "Real shared-expert demand and hard quotas limit the attainable fraction range; "
        "this is not an all-local versus all-remote experiment. Global expert arithmetic is "
        "fixed but per-rank compute distribution can change, so load CV remains a relevant "
        "covariate, and these results do not isolate link hardware latency.", "",
        "| Ranks | Event | Active experts | Achieved remote-fraction range | Median MoE latency range (ms) |",
        "|---|---|---:|---:|---:|"]
    for world in (8, 4):
        for event in ("low", "median", "high"):
            selected = [r for r in stage_a if r["world"] == world and r["event"] == event]
            fractions = [r["remote_pair_fraction"] for r in selected]
            seconds = [r["moe_seconds_median"] * 1000 for r in selected]
            text.append(f"| R{world} | {event} | {selected[0]['active_experts']} | {min(fractions):.3f}–{max(fractions):.3f} | {min(seconds):.3f}–{max(seconds):.3f} |")
    text += ["", "![Resident sensitivity](locality_sensitivity.png)", "",
        "[Cell summaries](local_remote_sensitivity.csv) and [all iterations](local_remote_iterations.csv) "
        "publish achieved fractions, route-local fractions, loads, payload and timing. "
        "MoE wall times are uninstrumented. Separate diagnostic CUDA-event intervals around "
        "dispatch/combine include stream/launch waits and are not isolated NCCL kernel durations. "
        "Routing metadata exchange and controller/solver time are excluded from these resident "
        "fixed-plan layer timings.", "",
        "## Relationships", "",
        "![Descriptive E2E relationships](e2e_relationships.png)", "",
        "TPOT is plotted against decode-only remote-pair fraction; generation panels use "
        "whole-generation counters. [Regression support](regression_support.csv) and "
        "[OLS coefficients](descriptive_regression.json) implement the prespecified descriptive "
        "model with remote pairs, expert H2D, controller time and rank CV. "
        "Predictors are correlated and workload/world effects remain. "
        f"There are {repeat_count} rows from {len(rows)} repeated conditions. "
        "No p-values, causal attribution or independent-sample generalization are claimed.", "",
        "## Two posthoc profiles", "",
        "Nsight Systems 2025.6.1 uses the previously validated configuration, with periodic stack "
        "snapshots disabled. Both profiles preserve every generated token (including after EOS), "
        "configuration, policy metric and physical fetch count from uninstrumented repeat 0. "
        "All observed large expert H2D transfers are 9 MiB and actual expert H2D equals logical fetch "
        "bytes on all 16 rank/cell combinations. No full-expert D2D remains.", "",
        "| R8/B8 policy | Actual expert H2D (GiB) | H2D/GEMM overlap (% of rank-summed H2D intervals) | Max-rank NCCL kernel union (s) |",
        "|---|---:|---:|---:|"]
    for row in profiles["rows"]:
        text.append(f"| {row['cell']} | {row['actual_expert_h2d_bytes'] / 2**30:.3f} | {row['h2d_overlap_fraction'] * 100:.2f}% | {row['max_rank_nccl_kernel_union_ms'] / 1000:.3f} |")
    text += ["", "[Profile summary](profile_summary.csv) and [rank receipts](profile_rank_receipts.csv) "
        "also include controller CPU wall time and gaps without recorded GPU kernel/copy/memset "
        "activity. NCCL kernels include device-side waits for peers; an active polling kernel "
        "is not counted as idle. GEMM overlap includes dense and expert matrix kernels. "
        "These overlapping diagnostic intervals must not be stacked or substituted for primary "
        "uninstrumented TPOT/E2E. Giant raw traces remain on the server.", "",
        "## Validation and stop", "",
        "[24 CPU tests](cpu_tests.txt) pass. The [binary reuse audit](baseline_reuse_audit.json) "
        "reconstructs the original validated runtime fingerprint using the current compiled binary. "
        "Prior R1/R4/R8 native and slot-view parity gates are reused with "
        "the unchanged compute/cache/collective implementation. New real-event incidence "
        "checks independently validate locality counters; actual submitted Stage-A tensors "
        "match dispatch+return byte formulas. All five repeats have identical full token "
        f"sequences and semantic/cache counters in every rank. All {rank_count} Stage-B physical-fetch "
        "receipts pass. [Validation](validation.json) includes source receipt hashes and the "
        "runtime fingerprint; [profile transfer audit](profile_transfer_audit.json) binds observed H2D.", "",
        "Submitted peer bytes exclude self traffic, router/count all-gather metadata and wire "
        "protocol overhead. Both full-generation and decode-only accounting are preserved. "
        "The OS exposes only NUMA node 0: no physical remote-NUMA or PCIe-only conclusion follows. "
        "Two warmup calls and five sequential repeats do not remove all clock, allocator or "
        "temporal variability. The optional 128-step confirmation was not run; the primary "
        "study and exactly two posthoc profiles define this execution's scope.", "",
        "**Stopped for owner review.** No timing-based coefficient tuning, replication, migration, "
        "quota change or additional policy search was performed.", ""]
    (root / "RESULTS.md").write_text("\n".join(text))
    (root / "acceptance.json").write_text(json.dumps(dict(status="PASS" if passed else "NEGATIVE_RESULT",
        directional_tpot_and_locality_gate=passed, anchors=anchors,
        validation_passed=True, actual_h2d_gate_passed=True, primary_repeats=5,
        scope="Prespecified directional gate, not a statistical significance test"), indent=2) + "\n")


if __name__ == "__main__":
    main()
