#!/usr/bin/env python3
"""Static study figures and descriptive comparisons; never selects policies."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(root / f"{name}.svg", bbox_inches="tight")
    path = root / f"{name}.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def table(root, name, rows):
    with (root / f"{name}.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    assert json.loads((root / "validation.json").read_text())["status"] == "PASS"
    b = json.loads((root / "e2e_repeats.json").read_text())["rows"]
    a = json.loads((root / "local_remote_sensitivity.json").read_text())["rows"]
    policies = ("random", "hungarian_current", "hungarian_same_path")
    labels = ("Random", "Hungarian current", "Hungarian same+path")
    colors = ("#586f7c", "#db9d47", "#167d8d")
    available = {(row["world"], row["local_batch"]) for row in b}
    cells = [cell for cell in ((8, 8), (8, 4), (8, 16), (8, 32), (4, 8)) if cell in available]
    summaries = []
    numeric = [key for key in b[0] if isinstance(b[0][key], (int, float)) and key not in ("world", "local_batch", "global_batch", "repeat")]
    for world, batch in cells:
        for policy in policies:
            rows = [row for row in b if (row["world"], row["local_batch"], row["policy"]) == (world, batch, policy)]
            assert len(rows) == 5
            entry = dict(world=world, local_batch=batch, global_batch=world * batch, policy=policy)
            for key in numeric:
                values = [row[key] for row in rows]
                entry.update({key + "_median": statistics.median(values), key + "_min": min(values), key + "_max": max(values)})
            summaries.append(entry)
    table(root, "e2e_summary", summaries)
    comparisons = []
    for world, batch in cells:
        rows = [r for r in summaries if (r["world"], r["local_batch"]) == (world, batch)]
        baseline = next(r for r in rows if r["policy"] == "random")
        for policy in policies[1:]:
            changed = next(r for r in rows if r["policy"] == policy)
            comparisons.append(dict(world=world, local_batch=batch, policy=policy,
                tpot_speedup=baseline["tpot_seconds_median"] / changed["tpot_seconds_median"],
                e2e_reduction=1 - changed["generation_seconds_median"] / baseline["generation_seconds_median"],
                remote_reduction=1 - changed["remote_pairs_median"] / baseline["remote_pairs_median"],
                remote_fraction_change=changed["remote_pair_fraction_median"] - baseline["remote_pair_fraction_median"],
                fetch_change=changed["fetches_median"] / baseline["fetches_median"] - 1))
    table(root, "comparisons", comparisons)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, len(cells), figsize=(4.3 * len(cells), 7), squeeze=False)
    for column, (world, batch) in enumerate(cells):
        for rowidx, metric in enumerate(("tpot_seconds", "generation_seconds")):
            ax = axes[rowidx, column]
            for x, policy in enumerate(policies):
                records = [r for r in b if (r["world"], r["local_batch"], r["policy"]) == (world, batch, policy)]
                values = [r[metric] for r in records]
                median = statistics.median(values)
                ax.scatter(np.linspace(x - .08, x + .08, 5), values, color=colors[x], s=22)
                ax.plot([x - .23, x + .23], [median, median], color=colors[x], lw=3)
                ax.vlines(x, min(values), max(values), color=colors[x], alpha=.5)
            ax.set_xticks(range(3), ["Random", "Current", "Same+path"])
            ax.set_ylabel("TPOT (s)" if rowidx == 0 else "Generation (s)")
            ax.grid(axis="y", alpha=.2)
            ax.set_ylim(bottom=0)
        axes[0, column].set_title(f"R{world} / local B{batch} / global B{world * batch}")
    fig.suptitle("Physical offloading: all five uninstrumented repeats", y=1.01)
    fig.text(.5, -.025, "cache30 • alpha=.25 / eta=.5 • prefill + 64 decode forwards • median bars and full ranges", ha="center")
    fig.tight_layout()
    save(fig, root, "e2e_timing")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, world in zip(axes, (8, 4)):
        for event, color in zip(("low", "median", "high"), colors):
            rows = sorted([r for r in a if r["world"] == world and r["event"] == event], key=lambda r: r["remote_pair_fraction"])
            x = [r["remote_pair_fraction"] for r in rows]
            y = [r["moe_seconds_median"] * 1000 for r in rows]
            ax.plot(x, y, "o-", color=color, label=f"{event} active experts")
            ax.vlines(x, [r["moe_seconds_min"] * 1000 for r in rows], [r["moe_seconds_max"] * 1000 for r in rows], color=color, alpha=.4)
        ax.set(title=f"R{world} / local B8", xlabel="Remote token-rank pair fraction", ylabel="Resident MoE latency (ms)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Fixed real events, resident experts: 20 iterations per owner map")
    fig.text(.5, -.03, "Same tokens, experts and hard quotas • median with min/max • no timed expert fetches • heuristic owner maps", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, root, "locality_sensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for cellidx, (world, batch) in enumerate(cells):
        marker = ("o", "s", "^", "D", "v")[cellidx]
        for policy, label, color in zip(policies, labels, colors):
            rows = [r for r in b if (r["world"], r["local_batch"], r["policy"]) == (world, batch, policy)]
            for ax, xkey, ykey, scale in ((axes[0], "decode_remote_pair_fraction", "tpot_seconds", 1),
                                          (axes[1], "remote_pair_fraction", "generation_seconds", 1),
                                          (axes[2], "physical_expert_h2d_bytes", "generation_seconds", 2**30)):
                ax.scatter([r[xkey] / scale for r in rows], [r[ykey] for r in rows], marker=marker,
                           color=color, alpha=.7, s=30, label=f"R{world}/B{batch} {label}")
    axes[0].set(xlabel="Decode remote pair fraction", ylabel="TPOT (s)")
    axes[1].set(xlabel="Remote pair fraction", ylabel="Generation (s)")
    axes[2].set(xlabel="Expert H2D (GiB; fetch accounting)", ylabel="Generation (s)")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[2].legend(fontsize=6, loc="best")
    fig.suptitle("End-to-end relationships (descriptive; generation panels include prefill)")
    fig.tight_layout()
    save(fig, root, "e2e_relationships")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key, title, scale in zip(axes.flat,
            ("physical_expert_h2d_bytes", "peer_payload_tx_bytes", "controller_seconds", "rank_token_cv"),
            ("Expert H2D (GiB; fetch accounting)", "Submitted dispatch+return (GiB)", "Controller wall time (s; max rank)", "Mean event rank token CV"),
            (2**30, 2**30, 1, 1)):
        for offset, policy, label, color in ((-.17, "random", "Random", colors[0]), (.17, "hungarian_same_path", "Same+path", colors[2])):
            rows = [next(r for r in summaries if (r["world"], r["local_batch"], r["policy"]) == (*cell, policy)) for cell in cells]
            values = [r[key + "_median"] / scale for r in rows]
            ax.bar(np.arange(len(cells)) + offset, values, width=.32, label=label, color=color)
        ax.set_xticks(range(len(cells)), [f"R{w}/B{batch}" for w, batch in cells])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("System metrics: Random vs frozen same+path (medians)")
    fig.tight_layout()
    save(fig, root, "system_metrics")
    # Scale predictors for readability. Repeats share semantic predictors;
    # coefficients and R² are descriptive, without p-values or causal claims.
    predictors = np.array([[r["remote_pairs"] / 1e6, r["physical_expert_h2d_bytes"] / 2**30,
                            r["controller_seconds"], r["rank_token_cv"]] for r in b])
    y = np.array([r["generation_seconds"] for r in b])
    design = np.column_stack((np.ones(len(b)), predictors))
    coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    fit = design @ coefficients
    regression = dict(n=len(b), distinct_conditions=len(summaries), matrix_rank=int(rank),
        coefficients=dict(zip(("intercept_seconds", "remote_pairs_per_million", "expert_h2d_gib", "controller_seconds", "rank_cv"), coefficients.tolist())),
        r_squared=float(1 - np.sum((y - fit)**2) / np.sum((y - y.mean())**2)),
        condition_number=float(singular[0] / singular[-1]),
        scope="Descriptive pooled OLS, not causal or inferential; repeated conditions and omitted workload/world effects limit interpretation")
    (root / "descriptive_regression.json").write_text(json.dumps(regression, indent=2) + "\n")
    table(root, "regression_support", [dict(world=r["world"], batch=r["local_batch"], policy=r["policy"], repeat=r["repeat"],
          remote_pairs_million=float(x[0]), h2d_gib=float(x[1]), controller_seconds=float(x[2]), rank_cv=float(x[3]),
          generation_seconds=float(actual), fitted_seconds=float(predicted)) for r, x, actual, predicted in zip(b, predictors, y, fit)])


if __name__ == "__main__":
    main()
